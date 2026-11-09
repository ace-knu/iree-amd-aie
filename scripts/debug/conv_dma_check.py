#!/usr/bin/env python3
"""Element-level check of a native conv dispatch's data movement (npu4).

Reads the IR dumped after iree-amdaie-assign-channels (the `--index`-th such
dump in the log, default the first) and composes, for every core, the access
patterns element by element:
  input  : core L1 <- memtile piece <- L3 input
  filter : core L1 <- memtile piece <- L3 filter
  output : core L1 -> memtile piece -> L3 output
The control code is interpreted (scf.for with constant bounds, affine.apply,
L3 transfers with SSA offsets); the L3 stream of a memtile piece, cut into
piece-sized fills, gives one L1 fill each (with the repetition counts checked
below), and the fills are paired by block: output fill b of a core goes with
its input and filter fills b*K .. b*K+K-1 (K input-channel chunks per block,
IC a whole number of them).
For every block of every core it then checks the conv's semantics: the output
fill is one [oh, ow, oc] block of one chunk n, each input fill is that
block's halo window of the chunk for channel chunk k, and each filter fill is
the block's output channels for every tap and channel chunk k. Finally the
cores together must write every output element exactly once, and the memtile
must drain to L3 nothing a core did not write: per piece, the cores' writes
cover exactly what the drain reads, and all drains together write exactly the
output box once. It also checks that no DMA channel serves two connections.

Around that:
  * every L3 transfer stays inside the memref of its logical objectFifo, and
    the transfers of each role (input, filter, output) all use one binding,
    three distinct ones of the role's shape (an input and an output can share
    a type: a conv writing into the interior of the next layer's padded
    input); the input and filter bindings are ReadOnly and the output's is
    not; and each transfer's base, the reinterpret_cast offset (0 without
    one) that becomes the BD address, is where the subspan's offset puts the
    operand in its binding, one base per role;
  * nothing moves outside the composed paths: each core has one output and
    two input connections, every DMA connection is on an input, filter or
    output path, and every L3 transfer belongs to a role;
  * the repetition count (leading stride-0 dims of a circular transfer) is
    the same on every side of a memtile piece (its L3 fill or drain and each
    core's read or write): the lowering turns only an objectFifo's common
    repetition count (getRepetitionCount, the minimum over its DMA users)
    into lock cycles, and a larger count on one side becomes extra BDs
    inside one cycle that replay the fill (AMDAIELowerToAIE
    createDMABlocks). An L1 objectFifo's only DMA user is its own side of
    the transfer, so that side is folded on its own and not compared. The
    count is the number of lock cycles in one round of the BD chain, not the
    number of fills. A piece that legitimately replays one L3 fill (reuse) is
    not modeled and fails;
  * the output's origin in its buffer (its smallest element written) must
    leave room for the output, must be 0 when the buffer has the output's
    shape, and must equal `--origin h,w,c` when that is given (the interior
    offset of a padded buffer is not in this dump). It is always printed.
Any inconsistency (a transfer whose element count does not match its peer, an
L1 position never filled, a pattern that does not fit) is a FAIL. Locks are
not modeled (see aie_audit.py), and nothing is executed: a PASS says the DMA
access patterns move the right elements, not that the values are right.

Shapes are taken from the L3 buffers: input [N][H+KH-1][W+KW-1][IC], filter
[OC][KH][KW][IC] (FHWC) or [KH][KW][IC][OC] (HWCF, --hwcf), output [N][H][W][OC]
of any element type, or a larger buffer the output is written into at an
interior offset (the next layer's padded input); the output is the L3 buffer
a connection writes, and the filter is the read buffer that fits the other as
the input for that output (any kernel size). N may be absent (a batch-folded
conv). Based on the T0.7 skeptic's e2e_check.py and the multi-block extension
written in the 11-08 review.

usage: conv_dma_check.py <compile stderr with assign-channels dumps> [--hwcf] [--index K]
                         [--origin H,W,C]
"""
import argparse
import re
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--hwcf", action="store_true")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--origin", help="expected output origin H,W,C in its buffer")
    a = ap.parse_args()
    hwcf, index = a.hwcf, a.index
    want_origin = tuple(int(x) for x in a.origin.split(",")) if a.origin else None
    raw = open(a.log, encoding="utf-8", errors="replace").read()
    heads = list(re.finditer(r"^// -----// IR Dump After AMDAIEAssignChannels", raw, re.M))
    if index >= len(heads):
        sys.exit(f"no assign-channels dump #{index} in the log ({len(heads)} found)")
    start = heads[index].end()
    nxt = raw.find("// -----// IR Dump", start)
    text = raw[start: nxt if nxt > 0 else len(raw)]
    const = {m.group(1): int(m.group(2)) for m in re.finditer(r"(%[\w.]+) = arith\.constant (-?\d+) : index", text)}
    tiles = {m.group(1): (const[m.group(2)], const[m.group(3)]) for m in re.finditer(r"(%[\w.]+) = amdaie\.tile\((%[\w.]+), (%[\w.]+)\)", text)}
    MR = r"memref<(?:[^<>]|<[^<>]*>)*>"
    lofs = {}
    for m in re.finditer(r"(%[\w.]+) = amdaie\.logicalobjectfifo\.from_memref (%[\w.]+), \{([^}]*)\} : (" + MR + r") -> !amdaie\.logicalobjectfifo<(" + MR + ")", text):
        lofs[m.group(1)] = dict(tiles=[tiles[t.strip()] for t in m.group(3).split(",") if t.strip()], orig=m.group(4), src=m.group(2))
    for m in re.finditer(r"(%[\w.]+) = amdaie\.logicalobjectfifo\.placeholder\{([^}]*)\} : !amdaie\.logicalobjectfifo<(" + MR + ")>", text):
        lofs[m.group(1)] = dict(tiles=[tiles[t.strip()] for t in m.group(2).split(",") if t.strip()], orig=m.group(3))
    # The binding behind an L3 memref: through assume_alignment and the
    # reinterpret_cast of a binding at an offset to the subspan. The cast's
    # static offset is the BD base address (AMDAIEControlCodeLowering), and
    # must be where the subspan puts the operand in its binding.
    src_of = {m.group(1): m.group(2) for m in re.finditer(
        r"(%[\w.]+) = memref\.(?:assume_alignment|reinterpret_cast) (%[\w.]+)", text)}
    cast_offset = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"(%[\w.]+) = memref\.reinterpret_cast %[\w.]+ to offset: \[(\d+)\]", text)}
    subspan = {}
    for m in re.finditer(r"(%[\w.]+) = hal\.interface\.binding\.subspan [^\n]*? binding\((\d+)\)"
                         r" alignment\(\d+\) offset\((%[\w.]+)\)(?: flags\(([^)]*)\))? : (" + MR + ")", text):
        subspan[m.group(1)] = dict(binding=int(m.group(2)), offset=const.get(m.group(3)),
                                   flags=m.group(4) or "", ty=m.group(5))

    def l3_origin(v):
        """(binding, cast offset in elements, the subspan) behind memref `v`."""
        seen, off = set(), 0
        while v in src_of and v not in seen:
            seen.add(v)
            if v in cast_offset and not off:
                off = cast_offset[v]
            v = src_of[v]
        sub = subspan.get(v)
        return (sub["binding"] if sub else None), off, sub
    conns = {}
    for m in re.finditer(r"(%[\w.]+) = amdaie\.connection\((%[\w.]+) \{([^}]*)\}, (%[\w.]+) \{([^}]*)\}\)", text):
        conns[m.group(1)] = dict(tgt=m.group(2), src=m.group(4),
                                 chans=[x.strip() for x in (m.group(3) + "," + m.group(5)).split(",") if x.strip()])
    chans = {}
    for m in re.finditer(r"(%[\w.]+) = amdaie\.channel\((%[\w.]+), (\d+), port_type = (\w+), "
                         r"direction = (\w+)\)", text):
        chans[m.group(1)] = (tiles[m.group(2)], int(m.group(3)), m.group(4), m.group(5))
    users = {}
    for name, c in conns.items():
        for ch in c["chans"]:
            if ch in chans and chans[ch][2] == "DMA":
                users.setdefault(chans[ch][:2] + (chans[ch][3],), set()).add(name)
    shared = {k: v for k, v in users.items() if len(v) > 1}

    def shape(ty):
        mm = re.match(r"memref<([0-9x]+)x(\w+)(?:,.*)?>$", ty)
        return [int(x) for x in mm.group(1).split("x")] if mm else None

    def elem_bytes(ty):
        mm = re.match(r"memref<(?:[0-9]+x)*(\w+?)(\d+)(?:,.*)?>$", ty)
        return int(mm.group(2)) // 8 if mm else None
    AP = r"\[([^\]]*)\] \[([^\]]*)\] \[([^\]]*)\]"
    cc = text.find("amdaie.controlcode")
    body = text[cc:]
    circ = {}
    for m in re.finditer(r"amdaie\.npu\.circular_dma_cpy_nd (%[\w.]+)\(" + AP + ", " + AP + r"\)", body):
        f = lambda x: [int(v) for v in x.split(",")] if x.strip() else []
        circ[m.group(1)] = dict(tgt=(f(m.group(2)), f(m.group(3)), f(m.group(4))), src=(f(m.group(5)), f(m.group(6)), f(m.group(7))))
    # Interpret the control code: loops, affine.apply, dma_cpy_nd with SSA offsets.
    lines = body.splitlines()
    stream_src, stream_tgt = {}, {}
    l3_lofs = {}
    bad = []

    def val(tok, env):
        tok = tok.strip()
        if re.fullmatch(r"-?\d+", tok): return int(tok)
        if tok in env: return env[tok]
        if tok in const: return const[tok]
        raise ValueError("unknown value " + tok)

    def enum(offs, sizes, strides):
        if not strides: return np.zeros(0, dtype=np.int64)
        base = sum(o * s for o, s in zip(offs, strides))
        a = np.array([base], dtype=np.int64)
        for n, s in zip(sizes, strides):
            a = (a[:, None] + np.arange(n, dtype=np.int64)[None, :] * s).reshape(-1)
        return a

    def run(i, env, end_depth):
        # executes lines from i until the matching closing brace; returns index after it
        while i < len(lines):
            ln = lines[i].strip()
            mfor = re.match(r"scf\.for (%[\w.]+) = (%[\w.]+|\d+) to (%[\w.]+|\d+) step (%[\w.]+|\d+) \{", ln)
            if mfor:
                lb, ub, st = val(mfor.group(2), env), val(mfor.group(3), env), val(mfor.group(4), env)
                j_end = None
                for iv in range(lb, ub, st):
                    env2 = dict(env); env2[mfor.group(1)] = iv
                    j_end = run(i + 1, env2, end_depth + 1)
                if j_end is None:  # zero-trip: skip body
                    depth = 1; j = i + 1
                    while depth:
                        s = lines[j].strip()
                        if s.endswith("{"): depth += 1
                        if s == "}": depth -= 1
                        j += 1
                    j_end = j
                i = j_end; continue
            if ln == "}":
                return i + 1
            ma = re.match(r"(%[\w.]+) = affine\.apply affine_map<\(([^)]*)\) -> \((.*)\)>\(([^)]*)\)", ln)
            if ma:
                dims = [d.strip() for d in ma.group(2).split(",")]
                args = [val(x, env) for x in ma.group(4).split(",")]
                expr = ma.group(3).replace("floordiv", "//").replace("mod", "%")
                assert re.fullmatch(r"[\sd0-9+\-*/%()]*", expr), expr
                env[ma.group(1)] = eval(expr, {}, dict(zip(dims, args)))
                i += 1; continue
            ms = re.match(r"%[\w.]+ = amdaie\.npu\.dma_cpy_nd async_source (%[\w.]+)\(\[\] \[\] \[\], (%[\w.]+)\[([^\]]*)\] \[([^\]]*)\] \[([^\]]*)\]\)", ln)
            mt = re.match(r"%[\w.]+ = amdaie\.npu\.dma_cpy_nd async_target (%[\w.]+)\((%[\w.]+)\[([^\]]*)\] \[([^\]]*)\] \[([^\]]*)\], \[\] \[\] \[\]\)", ln)
            for mm, store in ((ms, stream_src), (mt, stream_tgt)):
                if mm:
                    offs = [val(x, env) for x in mm.group(3).split(",")]
                    sizes = [int(x) for x in mm.group(4).split(",")]
                    strides = [int(x) for x in mm.group(5).split(",")]
                    e = enum(offs, sizes, strides)
                    lof = mm.group(2)
                    numel = int(np.prod(shape(lofs[lof]["orig"])))
                    if len(e) and (e.min() < 0 or e.max() >= numel):
                        bad.append(f"{mm.group(1)}: L3 transfer [{e.min()}, {e.max()}] outside its "
                                   f"{numel}-element {lof}")
                    store.setdefault(mm.group(1), []).append(e)
                    l3_lofs.setdefault(mm.group(1), set()).add(lof)
            if "amdaie.npu.dma_cpy_nd" in ln and not (ms or mt):
                raise ValueError("unparsed dma: " + ln)
            i += 1
        return i
    run(1, {}, 0)
    stream_src = {k: np.concatenate(v) for k, v in stream_src.items()}
    stream_tgt = {k: np.concatenate(v) for k, v in stream_tgt.items()}

    def lead0(p):
        k = 0
        while k < len(p[2]) and p[2][k] == 0: k += 1
        return k

    def fill_pattern(p):
        offs, sizes, strides = p
        k = lead0(p)
        return enum(offs[k:], sizes[k:], strides[k:]) + sum(o * s for o, s in zip(offs[:k], strides[:k]))

    def repetitions(p):
        return int(np.prod(p[1][:lead0(p)]))
    # L3 buffers by type and by role: connections write the output and read
    # the input and the filter.
    tgts = {c["tgt"] for c in conns.values()}
    srcs = {c["src"] for c in conns.values()}
    l3_read, l3_written = {}, {}
    for name, lf in lofs.items():
        if lf["tiles"] and lf["tiles"][0][1] == 0:
            if name in tgts:
                l3_written.setdefault(lf["orig"], []).append(name)
            if name in srcs:
                l3_read.setdefault(lf["orig"], []).append(name)
    shp = {ty: shape(ty) for ty in set(l3_read) | set(l3_written) if shape(ty) and len(shape(ty)) in (3, 4)}
    out_ty = [t for t in l3_written if t in shp]
    read = [t for t in l3_read if t in shp]

    # Roles by shape, for any kernel size: the filter is the read buffer that,
    # taken as [OC][KH][KW][IC] (or [KH][KW][IC][OC] with --hwcf), fits the
    # other read buffer as the input and the written one as the output.
    def fits(i, f):
        n_, hi_, wi_, ic_ = ([1] + shp[i])[-4:]
        if len(shp[f]) != 4: return False
        kh_, kw_, icf_, oc_ = shp[f] if hwcf else (shp[f][1], shp[f][2], shp[f][3], shp[f][0])
        nb_, hb_, wb_, cb_ = ([1] + shp[out_ty[0]])[-4:]
        return (icf_ == ic_ and nb_ == n_ and 1 <= hi_ - kh_ + 1 <= hb_
                and 1 <= wi_ - kw_ + 1 <= wb_ and oc_ <= cb_)
    pairs = [(i, f) for i in read for f in read if i != f and fits(i, f)] if len(out_ty) == 1 else []
    assert len(out_ty) == 1 and len(read) == 2 and len(pairs) == 1, (out_ty, read, pairs)
    in_ty, flt_ty = [pairs[0][0]], [pairs[0][1]]
    N, Hi, Wi, IC = ([1] + shp[in_ty[0]])[-4:]
    fs = shp[flt_ty[0]]
    KH, KW, ICf, OCf = fs if hwcf else (fs[1], fs[2], fs[3], fs[0])
    # The conv's output, and the buffer it is written into (the same shape, or
    # larger when it is written at an interior offset).
    OH, OW, OC = Hi - KH + 1, Wi - KW + 1, OCf
    Nb, Hb, Wb, Cb = ([1] + shp[out_ty[0]])[-4:]
    assert Nb == N and ICf == IC and Hb >= OH and Wb >= OW and Cb >= OC, (N, Hi, Wi, IC, fs, Nb, Hb, Wb, Cb)
    kind = lambda t: "shim" if t[1] == 0 else "memtile" if t[1] == 1 else "core"
    # The L3 connections each role uses, and the memtile pieces of the output
    # with the core writes into them.
    role_conns = {"input": set(), "filter": set(), "output": set()}
    piece_writes = {}
    on_path = set()

    def compose_in(core, l3names, role):
        for n1, c in conns.items():
            tgt = lofs[c["tgt"]]
            if core not in tgt["tiles"] or kind(tgt["tiles"][0]) != "core": continue
            piece = c["src"]
            up = [n for n, c2 in conns.items() if c2["tgt"] == piece]
            if len(up) != 1 or conns[up[0]]["src"] not in l3names: continue
            n2 = up[0]
            role_conns[role].add(n2)
            on_path.update((n1, n2))
            P1t, P1s = fill_pattern(circ[n1]["tgt"]), fill_pattern(circ[n1]["src"])
            P2 = fill_pattern(circ[n2]["tgt"])
            S = stream_src[n2]
            assert len(S) % len(P2) == 0, (n2, len(S), len(P2))
            nf = len(S) // len(P2)
            reps = (repetitions(circ[n1]["src"]), repetitions(circ[n2]["tgt"]))
            if len(set(reps)) != 1:
                bad.append(f"{core}: {role} piece repetitions (read, fill) {reps} differ")
            inv2 = np.full(int(max(P2.max(), P1s.max())) + 1, -1, dtype=np.int64)
            inv2[P2] = np.arange(len(P2))
            idx = inv2[P1s]
            assert (idx >= 0).all(), "L1 reads piece positions never filled"
            l1shape = shape(tgt["orig"])
            fills = []
            for f in range(nf):
                arr = np.full(int(np.prod(l1shape)), -1, dtype=np.int64)
                arr[P1t] = S[f * len(P2) + idx]
                fills.append(arr)
            return fills, l1shape
        return None, None

    def compose_out(core):
        for n1, c in conns.items():
            src = lofs[c["src"]]
            if core not in src["tiles"] or kind(src["tiles"][0]) != "core": continue
            piece = c["tgt"]
            down = [n for n, c2 in conns.items() if c2["src"] == piece]
            if len(down) != 1 or conns[down[0]]["tgt"] not in l3_written[out_ty[0]]: continue
            n2 = down[0]
            role_conns["output"].add(n2)
            on_path.update((n1, n2))
            P1s, P1t = fill_pattern(circ[n1]["src"]), fill_pattern(circ[n1]["tgt"])
            P2 = fill_pattern(circ[n2]["src"])
            piece_writes.setdefault(piece, (P2, []))
            T = stream_tgt[n2]
            assert len(T) % len(P2) == 0
            nb = len(T) // len(P2)
            reps = (repetitions(circ[n1]["tgt"]), repetitions(circ[n2]["src"]))
            if len(set(reps)) != 1:
                bad.append(f"{core}: output piece repetitions (write, drain) {reps} differ")
            inv2 = np.full(int(max(P2.max(), P1t.max())) + 1, -1, dtype=np.int64)
            inv2[P2] = np.arange(len(P2))
            idx = inv2[P1t]
            assert (idx >= 0).all(), "core writes piece positions never drained"
            l1shape = shape(src["orig"])
            fills = []
            for b in range(nb):
                arr = np.full(int(np.prod(l1shape)), -1, dtype=np.int64)
                arr[P1s] = T[b * len(P2) + idx]
                fills.append(arr)
            return fills, l1shape
        return None, None
    cores = sorted({t for t in tiles.values() if t[1] >= 2})
    written_all = []
    nfills = set()
    composed = {core: (compose_out(core), compose_in(core, l3_read[in_ty[0]], "input"),
                       compose_in(core, l3_read[flt_ty[0]], "filter")) for core in cores}
    # Where the output starts in its buffer: the first element any core writes.
    firsts = [int(o.min()) for (outs, _), _, _ in composed.values() if outs for o in outs if (o >= 0).all()]
    first = min(firsts) if firsts else 0
    orig_h, orig_w, orig_c = (first // (Wb * Cb)) % Hb, (first // Cb) % Wb, first % Cb
    origin = (orig_h, orig_w, orig_c)
    if orig_h + OH > Hb or orig_w + OW > Wb or orig_c + OC > Cb:
        bad.append(f"output origin {list(origin)} leaves no room for {OH}x{OW}x{OC} in a {Hb}x{Wb}x{Cb} buffer")
    if (Hb, Wb, Cb) == (OH, OW, OC) and origin != (0, 0, 0):
        bad.append(f"output origin {list(origin)} in a buffer of the output's shape")
    if want_origin is not None and origin != want_origin:
        bad.append(f"output origin {list(origin)} != expected {list(want_origin)}")
    # The binding behind each role's L3 transfers: one per role, three
    # distinct ones, each of its role's shape. The input and the filter are
    # ReadOnly bindings and the output is not, and every transfer's base (the
    # cast offset, 0 without one) is where the subspan puts the operand.
    role_ty = {"input": in_ty[0], "filter": flt_ty[0], "output": out_ty[0]}
    role_binding = {}
    for role, ns in role_conns.items():
        used = {lof for n in ns for lof in l3_lofs.get(n, ())}
        origins = {lof: l3_origin(lofs[lof].get("src")) for lof in used}
        bnds = {o[0] for o in origins.values()}
        if len(bnds) != 1 or None in bnds:
            bad.append(f"{role} transfers use bindings {sorted(bnds, key=str)}")
            continue
        role_binding[role] = next(iter(bnds))
        bases = {off for _, off, _ in origins.values()}
        if len(bases) != 1:
            bad.append(f"{role} transfers start at {len(bases)} places of binding {role_binding[role]}: "
                       f"elements {sorted(bases)}")
        for lof, (bnd, off, sub) in origins.items():
            if shape(lofs[lof]["orig"]) != shp[role_ty[role]]:
                bad.append(f"{role} transfer reads {lof} of shape {shape(lofs[lof]['orig'])}, "
                           f"not {shp[role_ty[role]]}")
            eb = elem_bytes(lofs[lof]["orig"])
            if sub["offset"] is None or eb is None or off * eb != sub["offset"]:
                bad.append(f"{role} transfer {lof} starts at element {off} of binding {bnd}, but its "
                           f"subspan puts the operand at byte {sub['offset']}")
            if ("ReadOnly" in sub["flags"]) != (role != "output"):
                bad.append(f"{role} binding {bnd} has flags({sub['flags']})")
    if len(set(role_binding.values())) != len(role_binding):
        bad.append(f"roles share a binding: {role_binding}")
    # Nothing moves outside the composed paths: each core has one output and
    # two input connections (input, filter), every connection is on a path,
    # and every L3 transfer belongs to a role.
    for core in cores:
        n_out = sum(1 for c in conns.values() if core in lofs[c["src"]]["tiles"])
        n_in = sum(1 for c in conns.values() if core in lofs[c["tgt"]]["tiles"])
        if (n_out, n_in) != (1, 2):
            bad.append(f"{core}: {n_out} output and {n_in} input connections, not 1 and 2")
    for name, c in conns.items():
        dma = any(ch in chans and chans[ch][2] == "DMA" for ch in c["chans"])
        if dma and name not in on_path:
            bad.append(f"DMA connection {name} is on no input, filter or output path")
    for name in sorted(set(stream_src) - role_conns["input"] - role_conns["filter"]):
        bad.append(f"L3 read by {name} outside the input and filter paths")
    for name in sorted(set(stream_tgt) - role_conns["output"]):
        bad.append(f"L3 write by {name} outside the output path")
    for core in cores:
        (outs, oshape), (inps, ishape), (flts, fshape) = composed[core]
        if outs is None or inps is None or flts is None:
            bad.append(f"{core}: missing connection"); continue
        th, tw, to = oshape[-3:]
        ICt = ishape[-1]
        if IC % ICt:
            bad.append(f"{core}: IC {IC} is not a whole number of {ICt}-channel chunks"); continue
        K = IC // ICt
        B = len(outs)
        nfills.add((B, K, len(inps), len(flts)))
        if len(inps) != B * K or len(flts) != B * K:
            bad.append(f"{core}: fills out {B} in {len(inps)} flt {len(flts)} K {K}"); continue
        assert ishape[-3:-1] == [th + KH - 1, tw + KW - 1], (ishape, oshape)
        t, w, o = np.meshgrid(np.arange(th), np.arange(tw), np.arange(to), indexing="ij")
        h2, w2, c2 = np.meshgrid(np.arange(th + KH - 1), np.arange(tw + KW - 1), np.arange(ICt), indexing="ij")
        if hwcf:
            kk, cc_, oo = np.meshgrid(np.arange(KH * KW), np.arange(ICt), np.arange(to), indexing="ij")
        else:
            oo, kk, cc_ = np.meshgrid(np.arange(to), np.arange(KH * KW), np.arange(ICt), indexing="ij")
        for b, out in enumerate(outs):
            if (out < 0).any():
                bad.append(f"{core}: out fill {b} has unwritten L1 positions"); continue
            v0 = int(out.min())
            # The block's origin in the buffer, then in the conv's output.
            n = v0 // (Hb * Wb * Cb)
            hb, wb, cb = (v0 // (Wb * Cb)) % Hb, (v0 // Cb) % Wb, v0 % Cb
            exp = (((n * Hb + hb + t) * Wb + wb + w) * Cb + cb + o).reshape(-1)
            oh0, ow0, oc0 = hb - orig_h, wb - orig_w, cb - orig_c
            if not np.array_equal(out, exp):
                bad.append(f"{core}: out fill {b} is not a block"); continue
            written_all.append(out)
            for k in range(K):
                inp = inps[b * K + k]
                expi = (((n * Hi + oh0 + h2) * Wi + ow0 + w2) * IC + k * ICt + c2).reshape(-1)
                if not np.array_equal(inp, expi):
                    bad.append(f"{core}: block {b} (n={n}, oh0={oh0}, ow0={ow0}, oc0={oc0}) input fill {k} mismatch"); break
                flt = flts[b * K + k]
                if hwcf:
                    expf = ((kk * IC + k * ICt + cc_) * OC + oc0 + oo).reshape(-1)
                else:
                    expf = (((oc0 + oo) * KH * KW + kk) * IC + k * ICt + cc_).reshape(-1)
                if not np.array_equal(flt, expf):
                    bad.append(f"{core}: block {b} filter fill {k} mismatch"); break
    allw = np.concatenate(written_all) if written_all else np.zeros(0, dtype=np.int64)
    total = N * OH * OW * OC
    uniq = np.unique(allw)
    nn, hh, ww, cc = np.meshgrid(np.arange(N), np.arange(OH), np.arange(OW), np.arange(OC), indexing="ij")
    box = np.sort((((nn * Hb + orig_h + hh) * Wb + orig_w + ww) * Cb + orig_c + cc).reshape(-1))
    full = len(allw) == total and len(uniq) == total and np.array_equal(uniq, box)
    # What the memtile drains to L3 must be what the cores wrote: per piece the
    # cores' writes are exactly the drained positions, and all drains together
    # write the output box once.
    for piece, (P2, _) in piece_writes.items():
        writes = []
        for n, c in conns.items():
            if c["tgt"] != piece: continue
            src = lofs[c["src"]]
            if not src["tiles"] or kind(src["tiles"][0]) != "core":
                bad.append(f"piece {piece}: written by {n} from {src['tiles']}, not a core"); continue
            writes.append(fill_pattern(circ[n]["tgt"]))
        w_all = np.concatenate(writes) if writes else np.zeros(0, dtype=np.int64)
        if len(w_all) != len(P2) or not np.array_equal(np.unique(w_all), np.unique(P2)):
            bad.append(f"piece {piece}: drained positions {len(np.unique(P2))} != positions the cores write "
                       f"{len(np.unique(w_all))} (of {len(w_all)} writes)")
    drained = np.sort(np.concatenate([stream_tgt[n] for n in role_conns["output"]])) \
        if role_conns["output"] else np.zeros(0, dtype=np.int64)
    drained_ok = np.array_equal(drained, box)
    print(f"conv N={N} OH={OH} OW={OW} OC={OC} IC={IC} KH={KH} KW={KW} {'HWCF' if hwcf else 'FHWC'}; cores {len(cores)}; fills (B,K,in,flt) {sorted(nfills)}")
    print(f"output elements written {len(allw)}, distinct {len(uniq)}, expected {total} at {list(origin)} "
          f"of a {Hb}x{Wb}x{Cb} buffer, exact cover {full}, drained to L3 {len(drained)} exact {drained_ok}")
    print(f"bindings {role_binding}; DMA channels shared by connections: {len(shared)}")
    for x in bad[:12]: print("MISMATCH", x)
    ok = full and drained_ok and not bad and not shared
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, ValueError, KeyError, IndexError) as e:
        # A transfer that does not fit its peer, an L1 position never filled,
        # an unparsed op: the data movement cannot be the conv's.
        print(f"RESULT: FAIL ({type(e).__name__}: {e})")
        sys.exit(1)
