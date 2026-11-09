#!/usr/bin/env python3
"""Static tile calculator for native 3x3 convolutions on npu4 (AIE2P).

For each VGG-16 conv layer (or a shape given on the command line) it searches
core tiles of a native conv dispatch and checks them against the hardware
limits a lowering must respect, before any compiler pass is written:

  L1   d_in*(ceil(9*IC_c*OC_t*2) + ceil((t_b+2)*(W+2)*IC_c*2))
       + d_acc*ceil(t_b*W*OC_t*4) + d_out*ceil(t_b*W*OC_t*b_out)
       <= 65536 - stack (1024)
       where ceil() rounds a buffer to its start alignment (64 B for npu4 core
       buffers of 64 B or more, else 32 B; commit 33c20094) and the last term
       is an in-core epilogue output (bf16: b_out = 2).
  MAC per byte received by a core (L2 -> L1):
       t_b*W*OC_t*9 / (18*OC_t + 2*(t_b+2)*(W+2))   (independent of IC_c)
  plus BD dimensions, BD transfer length (multiple of 4 B), IC chunk loop
  iterations folded into a shim BD (<= 63), L3 contiguous run, per-memtile L2,
  DMA channel budget, and the 64 B alignment of kw-shifted vector reads.

Grids (one dispatch per layer, 8 columns x 4 rows on npu4):
  g2      OC blocks on columns, OH halo chunks on rows (recommended)
  g1      OH chunks on columns, OC blocks on rows
  single  one column of 4 cores, OH chunks on rows (vertical slice)

The tile chosen per layer and grid maximizes MAC/B, then the L3 contiguous
run, then IC_c. Everything here is arithmetic: nothing is compiled.

Examples:
  python3 scripts/debug/conv_tile_calc.py                 # VGG-16, g2, table
  python3 scripts/debug/conv_tile_calc.py --grid single --epilogue bf16
  python3 scripts/debug/conv_tile_calc.py --shape 28,28,512,512 --all
  python3 scripts/debug/conv_tile_calc.py --selftest
"""
import argparse
import json
import math
import sys

# npu4 (AIE2P) limits; see docs_Conv/1103_conv_pipeline_tasks.md section 3.
CORE_L1 = 65536
STACK = 1024
MEMTILE_L2 = 512 * 1024
NUM_COLS, NUM_ROWS = 8, 4
SHIM_INTER_MAX = 63          # shim BD inter-dimension iterations (C11)
MEMTILE_BD_MAX = 48          # C3
BD_SIZE_MAX = 1023           # per-dimension size limit (C2)
CORE_S2MM_BYTES_PER_CYCLE = 4  # estimate per channel, not measured on hardware
PEAK_MAC_PER_CYCLE = 512     # bf16 (bfp16) 8x8x8 per cycle

VGG16 = [  # name, H(=W), IC, OC; all 3x3, stride 1, SAME padding
    ("conv1_1", 224, 3, 64), ("conv1_2", 224, 64, 64),
    ("conv2_1", 112, 64, 128), ("conv2_2", 112, 128, 128),
    ("conv3_1", 56, 128, 256), ("conv3_2", 56, 256, 256), ("conv3_3", 56, 256, 256),
    ("conv4_1", 28, 256, 512), ("conv4_2", 28, 512, 512), ("conv4_3", 28, 512, 512),
    ("conv5_1", 14, 512, 512), ("conv5_2", 14, 512, 512), ("conv5_3", 14, 512, 512),
]


def align(nbytes):
    """Bytes a core buffer occupies with its start aligned (33c20094)."""
    a = 64 if nbytes >= 64 else 32
    return -(-nbytes // a) * a


def round_up(x, m):
    return -(-x // m) * m


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def l1_bytes(t_b, w, oc_t, ic_c, d_in=2, d_acc=2, d_out=0, b_out=2):
    """Core L1 bytes of one tile without the stack."""
    weight = align(9 * ic_c * oc_t * 2)
    act = align((t_b + 2) * (w + 2) * ic_c * 2)
    acc = align(t_b * w * oc_t * 4)
    out = align(t_b * w * oc_t * b_out) if d_out else 0
    return d_in * (weight + act) + d_acc * acc + d_out * out


def mac_per_byte(t_b, w, oc_t):
    return t_b * w * oc_t * 9 / (18 * oc_t + 2 * (t_b + 2) * (w + 2))


def kw_shift_report(w, ic_i=8):
    """Byte offsets of the kw-shifted and next-row reads in an L1 activation
    tile laid out [t_b+2][IC_c/8][W+2][8] (bf16). A vector read of 512 bits or
    more needs a 64 B aligned address on AIE2P (C12)."""
    kw_offsets = [kw * ic_i * 2 for kw in range(3)]
    row_stride = (w + 2) * ic_i * 2
    bad = [o for o in kw_offsets if o % 64] + ([row_stride] if row_stride % 64 else [])
    return {"kw_offsets_B": kw_offsets, "row_block_stride_B": row_stride,
            "aligned_64B": not bad}


def layer_geometry(h, w_img, ic, oc, grid, cols=NUM_COLS, rows=NUM_ROWS):
    """Padded sizes and how the dispatch splits over the array."""
    ic_p = round_up(ic, 8)
    ow_p = round_up(w_img, 8)
    if grid == "g2":
        return {"ic_p": ic_p, "ow_p": ow_p, "oc_col": oc // cols, "oh_split": rows,
                "cores": cols * rows, "cols": cols}
    if grid == "g1":
        return {"ic_p": ic_p, "ow_p": ow_p, "oc_col": oc // rows, "oh_split": cols,
                "cores": cols * rows, "cols": cols}
    if grid == "single":
        return {"ic_p": ic_p, "ow_p": ow_p, "oc_col": oc, "oh_split": rows,
                "cores": rows, "cols": 1}
    raise ValueError(grid)


def evaluate(h, w_img, ic, oc, grid, t_b, w, oc_t, ic_c, ic_l2, d_in, d_acc,
             d_out, b_out, layout):
    """Checks one candidate tile; returns a dict with metrics and violations."""
    g = layer_geometry(h, w_img, ic, oc, grid)
    oh_p = round_up(h, g["oh_split"] * t_b)
    viol = []
    l1 = l1_bytes(t_b, w, oc_t, ic_c, d_in, d_acc, d_out, b_out)
    if l1 > CORE_L1 - STACK:
        viol.append(f"L1 {l1} > {CORE_L1 - STACK}")
    # BD shapes: memtile -> core activation [t_b+2][IC_c/8][W+2][8], weight
    # [9][IC_c/8][OC_t][8] (FHWC or pre-packed constant); both 4-D at most.
    for name, dims in (("act", [t_b + 2, ic_c // 8, w + 2, 8]),
                       ("weight", [9, ic_c // 8, oc_t, 8])):
        if any(d > BD_SIZE_MAX for d in dims):
            viol.append(f"{name} BD dim > {BD_SIZE_MAX}: {dims}")
    if (ic_c * 2) % 4:
        viol.append(f"BD transfer {ic_c * 2} B not a multiple of 4 B")
    # IC chunk loop: shim transfers one IC_L2 chunk per iteration; the loop is
    # folded into the shim BD inter dimension only up to 63 iterations.
    chunks_l2 = g["ic_p"] // ic_l2
    if chunks_l2 > SHIM_INTER_MAX:
        viol.append(f"IC chunk loop {chunks_l2} > {SHIM_INTER_MAX} shim iterations")
    # L3 contiguous run of an activation/weight chunk read.
    if layout == "nhwc":
        l3_run = ic_l2 * 2
    else:  # chunk-first layout written by the host / pre-packed constants
        l3_run = (w + 2) * ic_l2 * 2
    # L2 per memtile: weight slab + activation chunk + output join, depth 2.
    l2 = 2 * (9 * ic_l2 * oc_t * 2) + 2 * ((t_b + 2) * (w + 2) * ic_l2 * 2) \
        + 2 * (NUM_ROWS * t_b * w * oc_t * 4)
    if l2 > MEMTILE_L2:
        viol.append(f"L2 {l2} > {MEMTILE_L2}")
    mpb = mac_per_byte(t_b, w, oc_t)
    feed = min(1.0, mpb * 2 * CORE_S2MM_BYTES_PER_CYCLE / PEAK_MAC_PER_CYCLE)
    blocks = (oh_p // (g["oh_split"] * t_b)) * (g["ow_p"] // w) * (g["oc_col"] // oc_t)
    macs_padded = oh_p * g["ow_p"] * oc * 9 * g["ic_p"]
    macs_real = h * w_img * oc * 9 * ic
    return {
        "grid": grid, "t_b": t_b, "W": w, "OC_t": oc_t, "IC_c": ic_c, "IC_L2": ic_l2,
        "d_in": d_in, "d_acc": d_acc, "d_out": d_out, "L1": l1, "L2": l2,
        "MAC_per_B": round(mpb, 2), "peak_fraction_est": round(feed, 3),
        "IC_chunks_L2": chunks_l2, "IC_chunks_L1": g["ic_p"] // ic_c,
        "L3_run_B": l3_run, "blocks": blocks, "pad_MAC_ratio": round(macs_padded / macs_real, 3),
        "kw_shift": kw_shift_report(w), "violations": viol,
    }


def search(h, w_img, ic, oc, grid, epilogue, layout, all_ok=False, one_level=False):
    g = layer_geometry(h, w_img, ic, oc, grid)
    d_in, d_acc, d_out, b_out = 2, 2, 0, 2
    if epilogue:  # accumulator depth 1, bf16/f32 output depth 2 (section 5.0)
        d_acc, d_out = 1, 2
        b_out = {"bf16": 2, "f32": 4}[epilogue]
    rows_per_chunk = round_up(h, g["oh_split"]) // g["oh_split"]
    cands = []
    for t_b in divisors(rows_per_chunk):
        for w in [d for d in divisors(g["ow_p"]) if d % 8 == 0]:
            for oc_t in [d for d in divisors(g["oc_col"]) if d % 8 == 0]:
                for ic_c in [d for d in divisors(g["ic_p"]) if d % 8 == 0]:
                    l2_choices = [ic_c] if one_level else [d for d in divisors(g["ic_p"]) if d % ic_c == 0]
                    for ic_l2 in l2_choices:
                        c = evaluate(h, w_img, ic, oc, grid, t_b, w, oc_t, ic_c, ic_l2,
                                     d_in, d_acc, d_out, b_out, layout)
                        cands.append(c)
    ok = [c for c in cands if not c["violations"]]
    if all_ok:
        return sorted(ok, key=lambda c: (-c["MAC_per_B"], -c["L3_run_B"], -c["IC_c"]))
    if not ok:
        return [min(cands, key=lambda c: (len(c["violations"]), c["L1"]))]
    # Prefer one-level chunks (IC_L2 == IC_c) when they are legal, since the
    # two-level form needs an extra L2 staging step in the pipeline.
    best = max(ok, key=lambda c: (c["MAC_per_B"], c["IC_L2"] == c["IC_c"] or c["L3_run_B"] >= 128,
                                  c["L3_run_B"], c["IC_c"]))
    return [best]


def selftest():
    # Measured L1 overflows of today's conv-decompose tile (t_b=1, W=8, OC_t=8,
    # whole IC in L1): docs_Conv/1103_conv_pipeline_tasks.md section 5.0.
    assert l1_bytes(1, 8, 8, 256, d_in=1, d_acc=1) == 52480
    assert l1_bytes(1, 8, 8, 256, d_in=2, d_acc=2) + STACK == 105984
    assert l1_bytes(1, 8, 8, 512, d_in=1, d_acc=1) + STACK == 105728
    assert 9 * 512 * 8 * 2 == 73728
    assert round(mac_per_byte(1, 8, 8), 1) == 2.8
    # The plan's G2 table (option A, today's lowering, all depth 2).
    for (t_b, w, oc_t, ic_c), l1, mpb in [((28, 16, 8, 8), 48256, 26.4), ((14, 16, 16, 16), 56320, 37.3),
                                         ((14, 8, 32, 16), 57344, 36.0), ((7, 32, 16, 16), 57472, 35.8),
                                         ((4, 16, 64, 8), 54656, 26.9)]:
        assert l1_bytes(t_b, w, oc_t, ic_c) == l1, (t_b, w, oc_t, ic_c, l1_bytes(t_b, w, oc_t, ic_c))
        assert round(mac_per_byte(t_b, w, oc_t), 1) == mpb
        # With an in-core bf16 epilogue (acc depth 1, output depth 2) the same.
        assert l1_bytes(t_b, w, oc_t, ic_c, d_acc=1, d_out=2) == l1
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", choices=["g2", "g1", "single"], default="g2")
    ap.add_argument("--epilogue", choices=["bf16", "f32"], help="in-core epilogue output type")
    ap.add_argument("--layout", choices=["nhwc", "chunked"], default="nhwc",
                    help="L3 layout of activations/weights: plain NHWC/FHWC or chunk-first")
    ap.add_argument("--shape", help="H,W,IC,OC of one 3x3 stride-1 SAME conv instead of VGG-16")
    ap.add_argument("--all", action="store_true", help="list every legal tile, best first")
    ap.add_argument("--one-level", action="store_true",
                    help="IC chunked once (L2 chunk = L1 chunk), as in a GEMV-style pipeline")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    layers = VGG16
    if a.shape:
        h, w, ic, oc = (int(x) for x in a.shape.split(","))
        layers = [(f"conv_{h}x{w}x{ic}x{oc}", h, ic, oc)] if h == w else None
        if layers is None:
            sys.exit("only square images are supported")
    out = []
    for name, h, ic, oc in layers:
        res = search(h, h, ic, oc, a.grid, a.epilogue, a.layout, a.all, a.one_level)
        for r in res:
            r["layer"] = name
        out += res
    if a.json:
        print(json.dumps(out, indent=1))
        return
    hdr = ("layer", "t_b", "W", "OC_t", "IC_c", "IC_L2", "L1", "MAC/B", "feed", "chunks", "L3run",
           "blocks", "padMAC", "kw64B", "violations")
    print(" | ".join(hdr))
    for r in out:
        print(" | ".join(str(x) for x in (
            r["layer"], r["t_b"], r["W"], r["OC_t"], r["IC_c"], r["IC_L2"], r["L1"], r["MAC_per_B"],
            r["peak_fraction_est"], r["IC_chunks_L2"], r["L3_run_B"], r["blocks"], r["pad_MAC_ratio"],
            "ok" if r["kw_shift"]["aligned_64B"] else "no", "; ".join(r["violations"]) or "-")))


if __name__ == "__main__":
    main()
