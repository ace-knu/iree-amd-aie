#!/usr/bin/env python3
"""Static audit of an AMD-AIE dispatch against npu4 hardware limits.

Reads one iree-compile stderr log that holds IR dumps at the points below and
checks, per dispatch, what a lowering must respect before it can run:

  After  iree-amdaie-assign-channels      DMA channels per tile (shim 2/2,
                                          memtile 6/6, core 2/2), connections
                                          and their target counts (broadcast)
  After  iree-amdaie-assign-npu-dma-bd-ids  shim transfers, on the dims the
                                          lowering folds them to (unit dims
                                          dropped, contiguous dims merged
                                          within the dim limits): dims, inter
                                          iterations (<= 64; a stride-0 repeat
                                          must be the outermost dim, <= 256),
                                          wrap/step in 32-bit words (wrap <=
                                          1023, a wrap field's WrapMax, or
                                          1024 on the highest of 3 intra dims,
                                          which has no field; step <= 2^20),
                                          contiguous L3 run; for
                                          elements under 4 B, whole words: the
                                          base offset (with the from_memref
                                          offset), outer strides and the
                                          innermost run, unit innermost stride
                                          (folded again with the innermost dim
                                          limited in words when the first fold
                                          is not whole words, as the lowering
                                          does)
  After  iree-amdaie-lower-to-aie          tiles, locks, flows, columns; a
                                          lock acquired more than 1 at a time
                                          (AcquireGreaterEqual N > 1), or a
                                          lock whose init exceeds the buffers
                                          its DMA BDs guard: N parties share
                                          one counting lock pair, which is
                                          racy once a party can run ahead into
                                          the next buffer
                                          (AMDAIEObjFifoBufferization.cpp)
  Before amdaie-assign-bd-ids             core/memtile BDs: count (memtile
                                          <= 48), dims (core 3, memtile 4),
                                          wrap/step per dim in 32-bit words
                                          (core <= 256 / 8192, memtile
                                          <= 1024 / 131072; aie-rt AIE2P
                                          tables), no zero stride, innermost
                                          stride 1, length, outer strides and
                                          innermost size in 4 B (as
                                          configureDMABD requires)
  After  amdaie-assign-buffer-addresses   L1 per core = max(address + size)
                                          <= 65536 (stack included), 64 B
                                          start alignment of buffers >= 64 B

Known failure diagnostics in the log ("allocated buffers exceeded available
memory", "could not find and assign a valid BD id", stack bound, channel
assignment) count as violations, and so does every other error diagnostic: a
dispatch that fails before its first dump leaves no dump group, only its
errors. (The bank-aware buffer allocator's errors that it recovers from with
the sequential scheme, as the warning right after them says, are counted
apart, not as violations: a buffer it cannot place across the banks, a stack
larger than a bank.) A missing
dump header is an error: an empty dump must never pass, and a dispatch
without one of the five dumps, or with one that holds no IR, fails (its
checks would be skipped). The log's compile exit code is not seen here: check
it as well.

Usage:
  python3 scripts/debug/aie_audit.py --print-flags      # flags to add to iree-compile
  python3 scripts/debug/aie_audit.py compile_stderr.log [--json] [--expect-cores 32]
"""
import argparse
import itertools
import json
import re
import sys
from collections import Counter, defaultdict

DUMP_FLAGS = [
    "--mlir-disable-threading",
    "--mlir-print-ir-after=iree-amdaie-assign-channels",
    "--mlir-print-ir-after=iree-amdaie-assign-npu-dma-bd-ids",
    "--mlir-print-ir-after=iree-amdaie-lower-to-aie",
    "--mlir-print-ir-before=amdaie-assign-bd-ids",
    "--mlir-print-ir-after=amdaie-assign-buffer-addresses",
    "--mlir-elide-elementsattrs-if-larger=16",
]
PASSES = {
    "AMDAIEAssignChannels": "channels",
    "AMDAIEAssignNpuDmaBdIds": "shim",
    "AMDAIELowerToAIE": "aie",
    "AMDAIEAssignBufferDescriptorIDsPass": "bds",
    "AMDAIEAssignBufferAddresses": "buffers",
}
DIAGNOSTICS = [
    "allocated buffers exceeded available memory",
    "could not find and assign a valid BD id",
    "An upper bound for the stack size",
    "no producer DMA channel available",
    "no consumer DMA channel available",
    "failed to legalize operation",
]
CORE_L1 = 65536
CHANNEL_LIMITS = {"shim": 2, "memtile": 6, "core": 2}
BD_DIMS = {"shim": 3, "memtile": 4, "core": 3}
MEMTILE_BD_MAX = 48
# Per-dimension wrap and step limits of a BD in 32-bit words: aie-rt accepts
# up to WrapMax + 1 and StepSizeMax + 1 of the AIE2P (npu4) tables in
# third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c (compute tile,
# memtile, shim). A dim's size is in elements; the innermost dim's wrap and the
# outer dims' steps are scaled to 32-bit words (mlir-aie configureBdInBlock).
BD_WRAP_MAX = {"core": 256, "memtile": 1024, "shim": 1024}
BD_STEP_MAX = {"core": 8192, "memtile": 131072, "shim": 1 << 20}
# aie-rt takes an iteration wrap up to IterWrapMax + 1 (it writes wrap - 1).
SHIM_ITER_MAX = 64
SHIM_REPEAT_MAX = 256
ELEM_BYTES = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i8": 1, "i16": 2, "f64": 8, "i64": 8}
HDR = re.compile(r"^// -----// IR Dump (After|Before) (\w+) \(([^)]*)\)")
ERROR_LINE = re.compile(r"^(?:\S.*: )?error: (.*)$")
NOTE_LINE = re.compile(r"^(?:\S.*: )?note: ")
BANK_FALLBACK = "Bank-aware scheme for buffer address assignment is failed. Try the basic sequential scheme."
# Shim BD dim limits as the lowering folds dims (DmaDimConfig::getMaxSizes):
# WrapMax per intra dim, the highest of 3 intra dims unbounded, IterWrapMax for
# the inter (iteration) dim.
SHIM_INTRA_DIMS = 3
SHIM_WRAP_MAX = 1023
SHIM_ITER_WRAP_MAX = 63


def count_errors(text):
    """(error diagnostics, recovered bank-aware allocation errors) in the log.
    An error the bank-aware allocator recovers from is followed, after its note
    lines, by the warning that it falls back to the sequential scheme (whose own
    failure is a separate error)."""
    lines = text.splitlines()
    errors, fallbacks = 0, 0
    for i, line in enumerate(lines):
        if not ERROR_LINE.match(line):
            continue
        j = i + 1
        while j < len(lines) and NOTE_LINE.match(lines[j]):
            j += 1
        if j < len(lines) and BANK_FALLBACK in lines[j]:
            fallbacks += 1
        else:
            errors += 1
    return errors, fallbacks


def shim_max_sizes(n):
    """DmaDimConfig::getMaxSizes for n shim dims."""
    maxs = [0] * n
    intra = min(SHIM_INTRA_DIMS, n)
    start = n - intra
    for i in range(start, n):
        maxs[i] = SHIM_WRAP_MAX
    if intra == SHIM_INTRA_DIMS:
        maxs[start] = float("inf")
    for i in range(max(start - 1, 0), start):
        maxs[i] = SHIM_ITER_WRAP_MAX
    return maxs


def fold_dims(sizes, strides, eb, innermost_in_words=False):
    """The dims a shim BD gets (AMDAIEControlCodeLowering): unit dims dropped,
    then foldLinearDims merges an outer dim into the dim inside it when they
    are contiguous and the merged size fits that dim's limit. The lowering
    folds a sub-word access a second time, limiting the innermost dim by its
    size in words, when the first fold leaves it not whole words."""
    dims = [(sz, st) for sz, st in zip(sizes, strides) if sz != 1]
    if not dims:
        return []
    maxs = shim_max_sizes(len(dims))
    new = [dims[-1]]
    for sz, st in reversed(dims[:-1]):
        psz, pst = new[-1]
        k = len(new) - 1
        merged = sz * psz
        limited = -(-merged * eb // 4) if innermost_in_words and k == 0 else merged
        if k < len(maxs) and limited <= maxs[len(maxs) - k - 1] and pst * psz == st:
            new[-1] = (merged, pst)
        else:
            new.append((sz, st))
    return list(reversed(new))


def sub_word_problems(folded, eb):
    """What makes a folded sub-word shim access not whole words (the base
    offset is checked apart)."""
    if not folded:
        return [f"shim transfer of one {eb}-byte element"]
    out = []
    isz, ist = folded[-1]
    if isz > 1 and ist != 1:
        out.append(f"shim innermost stride {ist} for {eb}-byte elements")
    if isz * eb % 4:
        out.append(f"shim innermost run {isz * eb} B not a multiple of 4")
    for sz, st in folded[:-1]:
        if sz > 1 and st != 0 and st * eb % 4:
            out.append(f"shim stride {st * eb} B not a multiple of 4")
    return out


def kind_of_row(row):
    return "shim" if row == 0 else "memtile" if row == 1 else "core"


def memref_bytes(ty):
    m = re.match(r"memref<([0-9x]*)(\w+)", ty)
    if not m:
        return None, None
    dims = [int(x) for x in m.group(1).split("x") if x]
    n = 1
    for d in dims:
        n *= d
    eb = ELEM_BYTES.get(m.group(2))
    return (n * eb if eb else None), eb


def split_dumps(text):
    """Returns [(direction, pass class, body)] in log order."""
    out, cur, buf = [], None, []
    for line in text.splitlines():
        m = HDR.match(line)
        if m:
            if cur:
                out.append((cur[0], cur[1], "\n".join(buf)))
            cur, buf = (m.group(1), m.group(2)), []
            continue
        if cur:
            buf.append(line)
    if cur:
        out.append((cur[0], cur[1], "\n".join(buf)))
    return out


def group_dispatches(dumps):
    """Groups dumps per dispatch: a pass seen again starts a new group."""
    groups, cur = [], {}
    for direction, cls, body in dumps:
        key = PASSES.get(cls)
        if key is None:
            continue
        if key in cur:
            groups.append(cur)
            cur = {}
        cur[key] = body
    if cur:
        groups.append(cur)
    return groups


def audit_channels(body, res):
    consts = {m.group(1): int(m.group(2)) for m in
              re.finditer(r"(%c\w*) = arith\.constant (-?\d+) : index", body)}
    tiles = {}
    for m in re.finditer(r"(%[\w.]+) = amdaie\.tile\((%\w+|\d+), (%\w+|\d+)\)", body):
        col = consts.get(m.group(2), None) if m.group(2).startswith("%") else int(m.group(2))
        row = consts.get(m.group(3), None) if m.group(3).startswith("%") else int(m.group(3))
        tiles[m.group(1)] = (col, row)
    used = defaultdict(set)
    for m in re.finditer(r"amdaie\.channel\((%[\w.]+), (\d+), port_type = (\w+), direction = (\w+)\)", body):
        t = tiles.get(m.group(1))
        if t is None or m.group(3) != "DMA":
            continue
        used[(t, m.group(4))].add(int(m.group(2)))
    per_kind = defaultdict(int)
    for (t, d), chans in used.items():
        kind = kind_of_row(t[1])
        per_kind[(kind, d)] = max(per_kind[(kind, d)], len(chans))
        if len(chans) > CHANNEL_LIMITS[kind] or max(chans) >= CHANNEL_LIMITS[kind]:
            res["violations"].append(f"{kind} tile {t} uses {d} channels {sorted(chans)} (limit {CHANNEL_LIMITS[kind]})")
    res["max_channels"] = {f"{k}.{d}": n for (k, d), n in sorted(per_kind.items())}
    targets = Counter()
    for m in re.finditer(r"amdaie\.connection\((%[\w.]+) \{([^}]*)\}, (%[\w.]+) \{([^}]*)\}\)", body):
        n_tgt = len([c for c in m.group(2).split(",") if c.strip()])
        targets[n_tgt] += 1
    res["connections_by_target_channels"] = dict(sorted(targets.items()))
    res["cores"] = len({t for t in tiles.values() if t[1] is not None and t[1] >= 2})


def audit_shim(body, res):
    runs, n_ops = Counter(), 0
    # The static offset of an L3 memref (a reinterpret_cast of a binding at an
    # offset, see AMDAIEConvertToDma) adds to the BD base address.
    base = {}
    for m in re.finditer(r"(%[\w.]+) = amdaie\.logicalobjectfifo\.from_memref [^:\n]*: (memref<[^\n]*?>) -> ", body):
        off = re.search(r"offset: (\d+)", m.group(2))
        base[m.group(1)] = int(off.group(1)) if off else 0
    consts = {m.group(1): int(m.group(2)) for m in
              re.finditer(r"(%[\w.]+) = arith\.constant (-?\d+) : index", body)}
    # Values an SSA offset of the control code can take: a constant, a loop
    # IV with constant bounds, or an affine.apply of those.
    ivs = {}
    for m in re.finditer(r"scf\.for (%[\w.]+) = (%[\w.]+|\d+) to (%[\w.]+|\d+) step (%[\w.]+|\d+)", body):
        b = [int(x) if x.isdigit() else consts.get(x) for x in m.group(2, 3, 4)]
        if None not in b:
            ivs[m.group(1)] = range(*b)
    applied = {}
    for m in re.finditer(r"(%[\w.]+) = affine\.apply affine_map<\(([^)]*)\) -> \((.*?)\)>\(([^)]*)\)", body):
        dims = [d.strip() for d in m.group(2).split(",")]
        args = [a.strip() for a in m.group(4).split(",")]
        expr = m.group(3).replace("floordiv", "//").replace("mod", "%")
        doms = [ivs.get(a, [consts[a]] if a in consts else None) for a in args]
        if None in doms or not re.fullmatch(r"[\sd0-9+\-*/%()]*", expr):
            continue
        applied[m.group(1)] = {eval(expr, {}, dict(zip(dims, c))) for c in itertools.product(*doms)}

    def values(tok):
        if re.fullmatch(r"-?\d+", tok):
            return {int(tok)}
        if tok in consts:
            return {consts[tok]}
        return set(ivs[tok]) if tok in ivs else applied.get(tok)

    unchecked = 0
    for m in re.finditer(r"amdaie\.npu\.dma_cpy_nd [^\n]*", body):
        line = m.group(0)
        n_ops += 1
        # Each side: %lof[offsets] [sizes] [strides]; skip empty "[] [] []" sides.
        for side in re.finditer(r"(%[\w.]+)\[([^\]]*)\] \[([^\]]*)\] \[([^\]]*)\]", line):
            sizes = [int(x) for x in side.group(3).split(",") if x.strip()]
            strides = [int(x) for x in side.group(4).split(",") if x.strip()]
            if not sizes:
                continue
            offsets = [x.strip() for x in side.group(2).split(",") if x.strip()]
            ty = re.search(r"logicalobjectfifo<(memref<[^>]*>)>", line)
            eb = memref_bytes(ty.group(1))[1] if ty else None
            # The BD gets the dims the lowering folds to (unit dims dropped,
            # contiguous dims merged within the dim limits); for elements
            # under 4 B that are not whole words that way, it folds again with
            # the innermost dim limited in words.
            real = fold_dims(sizes, strides, eb or 4)
            problems = []
            if eb and eb < 4:
                # The start's bytes mod 4, over every value an offset takes; an
                # offset on a dim whose stride is whole words cannot change it.
                rest = {base.get(side.group(1), 0) * eb % 4}
                for tok, st in zip(offsets, strides):
                    if rest is None or st * eb % 4 == 0:
                        continue
                    vs = values(tok)
                    rest = None if vs is None else {(r + v * st * eb) % 4 for r in rest for v in vs}
                if rest is None:
                    unchecked += 1
                elif rest != {0}:
                    res["violations"].append(f"shim base offset not a multiple of 4 B: {line[:160]}")
                problems = sub_word_problems(real, eb)
                if problems:
                    real = fold_dims(sizes, strides, eb, True)
                    problems = sub_word_problems(real, eb)
            for problem in problems:
                res["violations"].append(f"{problem}: {line[:160]}")
            # A shim BD holds 3 intra dimensions and one inter (iteration)
            # dimension, at most 63 iterations unless its stride is 0 (repeat,
            # at most 256).
            if len(real) > 4:
                res["violations"].append(f"shim transfer with {len(real)} folded dims: {line[:160]}")
            # A stride-0 dim can only be the repeat of the outermost
            # (iteration) dim: intra dims need a non-zero step.
            if any(st == 0 for _, st in real[1:]):
                res["violations"].append(f"shim stride-0 dim that is not the outermost: {line[:160]}")
            iter_dim = bool(real) and (len(real) == 4 or real[0][1] == 0)
            if iter_dim:
                lim = SHIM_REPEAT_MAX if real[0][1] == 0 else SHIM_ITER_MAX
                if real[0][0] > lim:
                    res["violations"].append(f"shim outer iterations {real[0][0]} > {lim}: {line[:160]}")
            intra = real[1:] if iter_dim else real
            for i, (sz, st) in enumerate(intra):
                for what, words, lim in bd_dim_words(i == len(intra) - 1 and st == 1, sz, st, eb, "shim"):
                    # A wrap field holds at most WrapMax: aie-rt takes WrapMax + 1
                    # but writes it as 0. The highest of 3 intra dims has no wrap
                    # field, yet aie-rt still takes at most WrapMax + 1 there.
                    if what == "wrap":
                        lim = SHIM_WRAP_MAX + (len(intra) - 1 - i == SHIM_INTRA_DIMS - 1)
                    if words > lim:
                        res["violations"].append(f"shim dim {what} {words} words > {lim}: {line[:160]}")
            if eb and strides and strides[-1] == 1:
                runs[sizes[-1] * eb] += 1
    res["shim_dma_ops"] = n_ops
    res["l3_contiguous_run_bytes"] = dict(sorted(runs.items()))
    if unchecked:
        res["shim_offsets_unchecked"] = unchecked


def audit_aie(body, res):
    tiles = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in
             re.finditer(r"(%[\w.]+) = aie\.tile\((\d+), (\d+)\)", body)}
    res["tiles"] = Counter(kind_of_row(r) for _, r in tiles.values())
    res["columns"] = len({c for c, _ in tiles.values()})
    locks = Counter()
    for m in re.finditer(r"aie\.lock\((%[\w.]+), \d+\)", body):
        t = tiles.get(m.group(1))
        if t:
            locks[kind_of_row(t[1])] += 1
    res["locks"] = dict(locks)
    res["flows"] = len(re.findall(r"aie\.flow\(", body))
    # Producers (or consumers) of one buffer that are not contiguous in-order
    # slices of it get a single counting lock pair instead of one pair each:
    # the other side then acquires the count of all parties at once.
    counting = Counter(int(n) for n in re.findall(
        r"aie\.use_lock\(%[\w.]+, AcquireGreaterEqual, (\d+)\)", body) if int(n) > 1)
    res["counting_lock_acquires"] = dict(sorted(counting.items()))
    if counting:
        res["violations"].append(
            f"{sum(counting.values())} lock acquires of N > 1 ({dict(counting)}): a shared counting "
            "lock pair joins or fans out several parties, racy when one can run ahead")
    # A shared pair whose other side is a core keeps every acquire at 1, but
    # its producer lock starts at parties x depth: more than the buffers its
    # DMA BDs guard. (One pair per party starts at the depth.)
    inits = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"(%[\w.]+) = aie\.lock\(%[\w.]+, \d+\) \{init = (\d+)", body)}
    guarded = defaultdict(set)
    blk_locks, blk_bufs = set(), set()
    for line in body.splitlines() + ["^end:"]:
        if re.match(r"\s*\^\w+:", line) or "aie.end" in line:
            for lk in blk_locks:
                guarded[lk] |= blk_bufs
            blk_locks, blk_bufs = set(), set()
            continue
        m = re.search(r"aie\.use_lock\((%[\w.]+),", line)
        if m:
            blk_locks.add(m.group(1))
        m = re.search(r"aie\.dma_bd\((%[\w.]+)", line)
        if m:
            blk_bufs.add(m.group(1))
    over = sorted((lk, inits[lk], len(b)) for lk, b in guarded.items() if b and inits.get(lk, 0) > len(b))
    res["locks_init_over_buffers"] = len(over)
    for lk, init, nb in over[:10]:
        res["violations"].append(f"lock {lk} starts at {init} but its DMA BDs guard {nb} buffers: a shared "
                                 "counting pair (one party can run ahead into the next buffer)")


def bd_dim_words(innermost, size, stride, eb, kind):
    """(what, value in 32-bit words, limit) to check for one BD dim: the wrap
    (size, scaled to words for the innermost dim) and, for an outer dim, the
    step (stride scaled to words). Unknown element size: elements as words."""
    scale = (eb or 4) / 4
    wrap = -(-size * scale // 1) if innermost else size
    out = [("wrap", int(wrap), BD_WRAP_MAX[kind])]
    if not innermost:
        out.append(("step", int(-(-stride * scale // 1)), BD_STEP_MAX[kind]))
    return out


def audit_bds(body, res):
    tiles = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in
             re.finditer(r"(%[\w.]+) = aie\.tile\((\d+), (\d+)\)", body)}
    buffers = {}
    for m in re.finditer(r"(%[\w.]+) = aie\.buffer\((%[\w.]+)\)[^:\n]*: (memref<[^>]*>)", body):
        buffers[m.group(1)] = m.group(3)
    per_tile = Counter()
    max_dims = defaultdict(int)
    cur_tile = None
    for line in body.splitlines():
        m = re.search(r"= aie\.(mem|memtile_dma|shim_dma)\((%[\w.]+)\)", line)
        if m:
            cur_tile = tiles.get(m.group(2))
            continue
        m = re.search(r"aie\.dma_bd\((%[\w.]+) : (memref<[^>]*>)(?:, (\d+), (\d+))?\)(.*)", line)
        if not m or cur_tile is None:
            continue
        kind = kind_of_row(cur_tile[1])
        per_tile[cur_tile] += 1
        attrs = m.group(5)
        dims = [(int(a), int(b)) for a, b in re.findall(r"<size = (\d+), stride = (\d+)>", attrs)]
        max_dims[kind] = max(max_dims[kind], len(dims))
        if len(dims) > BD_DIMS[kind]:
            res["violations"].append(f"{kind} {cur_tile} BD has {len(dims)} dims: {line.strip()[:160]}")
        _, eb = memref_bytes(m.group(2))
        for i, (size, stride) in enumerate(dims):
            if stride == 0:
                res["violations"].append(f"{kind} {cur_tile} BD stride 0")
            for what, words, lim in bd_dim_words(i == len(dims) - 1, size, stride, eb, kind):
                if words > lim:
                    res["violations"].append(f"{kind} {cur_tile} BD dim {i} {what} {words} words > {lim}")
        if dims and dims[-1][1] != 1 and eb and eb < 4:
            res["violations"].append(f"{kind} {cur_tile} BD innermost stride {dims[-1][1]} for {eb}-byte elements")
        # The BD counts strides and the innermost size in 32-bit words
        # (configureDMABD in iree_aie_configure.cc rejects anything else).
        if dims and eb:
            for i, (size, stride) in enumerate(dims[:-1]):
                if stride * eb % 4:
                    res["violations"].append(f"{kind} {cur_tile} BD dim {i} stride {stride * eb} B not a multiple of 4")
            if dims[-1][0] * eb % 4:
                res["violations"].append(f"{kind} {cur_tile} BD innermost size {dims[-1][0] * eb} B not a multiple of 4")
        ln = re.search(r"len = (\d+)", attrs)
        if ln and eb and (int(ln.group(1)) * eb) % 4:
            res["violations"].append(f"{kind} {cur_tile} BD length {int(ln.group(1)) * eb} B not a multiple of 4")
    mt = [n for t, n in per_tile.items() if kind_of_row(t[1]) == "memtile"]
    res["memtile_bds_max"] = max(mt) if mt else 0
    res["core_bds_max"] = max([n for t, n in per_tile.items() if kind_of_row(t[1]) == "core"] or [0])
    res["bd_dims_max"] = dict(max_dims)
    for t, n in per_tile.items():
        if kind_of_row(t[1]) == "memtile" and n > MEMTILE_BD_MAX:
            res["violations"].append(f"memtile {t} has {n} BDs > {MEMTILE_BD_MAX}")


def audit_buffers(body, res):
    tiles = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in
             re.finditer(r"(%[\w.]+) = aie\.tile\((\d+), (\d+)\)", body)}
    ends, misaligned = defaultdict(int), []
    for m in re.finditer(r"aie\.buffer\((%[\w.]+)\) \{([^}]*)\} : (memref<[^>]*>)", body):
        t = tiles.get(m.group(1))
        addr = re.search(r"address = (\d+)", m.group(2))
        nbytes, _ = memref_bytes(m.group(3))
        if t is None or addr is None or nbytes is None or kind_of_row(t[1]) != "core":
            continue
        a = int(addr.group(1))
        ends[t] = max(ends[t], a + nbytes)
        if nbytes >= 64 and a % 64:
            misaligned.append((t, a, nbytes))
    res["l1_max_end"] = max(ends.values()) if ends else 0
    for t, e in ends.items():
        if e > CORE_L1:
            res["violations"].append(f"core {t} L1 end {e} > {CORE_L1}")
    for t, a, n in misaligned[:10]:
        res["violations"].append(f"core {t} buffer of {n} B at address {a} not 64 B aligned")


def audit(text, expect_cores=None):
    dumps = split_dumps(text)
    groups = group_dispatches(dumps)
    report = {"dispatches": [], "diagnostics": {}, "violations": 0}
    for d in DIAGNOSTICS:
        n = text.count(d)
        if n:
            report["diagnostics"][d] = n
    n_err, n_bank = count_errors(text)
    if n_err:
        report["diagnostics"]["error diagnostics in the log"] = n_err
    report["bank_fallbacks"] = n_bank
    if not groups:
        report["error"] = "no recognized IR dump headers: add the --print-flags flags to iree-compile"
        report["violations"] = sum(report["diagnostics"].values())
        return report
    for g in groups:
        res = {"violations": [], "missing_dumps": sorted(set(PASSES.values()) - set(g))}
        for kind in res["missing_dumps"]:
            res["violations"].append(f"missing {kind} dump: its checks did not run")
        for kind, body in sorted(g.items()):
            if "func.func" not in body and "aie.device" not in body:
                res["violations"].append(f"{kind} dump holds no IR: its checks did not run")
        if "channels" in g:
            name = re.search(r"func\.func @([\w$.-]+)", g["channels"])
            res["name"] = name.group(1) if name else "?"
            audit_channels(g["channels"], res)
        if "shim" in g:
            audit_shim(g["shim"], res)
        if "aie" in g:
            audit_aie(g["aie"], res)
        if "bds" in g:
            audit_bds(g["bds"], res)
        if "buffers" in g:
            audit_buffers(g["buffers"], res)
        if res.get("cores") == 0:
            res["violations"].append("no core tiles")
        if expect_cores is not None and res.get("cores") != expect_cores:
            res["violations"].append(f"cores {res.get('cores')} != expected {expect_cores}")
        report["dispatches"].append(res)
        report["violations"] += len(res["violations"])
    report["violations"] += sum(report["diagnostics"].values())
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?")
    ap.add_argument("--print-flags", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--expect-cores", type=int)
    a = ap.parse_args()
    if a.print_flags:
        print(" ".join(DUMP_FLAGS))
        return 0
    if not a.log:
        ap.error("log file required")
    text = open(a.log, encoding="utf-8", errors="replace").read()
    rep = audit(text, a.expect_cores)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        if "error" in rep:
            print("ERROR:", rep["error"])
        for k, v in rep["diagnostics"].items():
            print(f"DIAGNOSTIC x{v}: {k}")
        if rep.get("bank_fallbacks"):
            print(f"recovered bank-aware allocation errors: {rep['bank_fallbacks']}")
        for i, r in enumerate(rep["dispatches"]):
            print(f"[{i}] {r.get('name', '?')}")
            for k in ("cores", "tiles", "columns", "max_channels", "connections_by_target_channels",
                      "locks", "flows", "memtile_bds_max", "core_bds_max", "bd_dims_max",
                      "l1_max_end", "shim_dma_ops", "l3_contiguous_run_bytes", "shim_offsets_unchecked",
                      "counting_lock_acquires", "locks_init_over_buffers", "missing_dumps"):
                if k in r:
                    print(f"    {k}: {dict(r[k]) if isinstance(r[k], Counter) else r[k]}")
            for v in r["violations"]:
                print("    VIOLATION:", v)
        print("RESULT:", "FAIL" if rep["violations"] or "error" in rep else "PASS",
              f"({rep['violations']} violations)")
    return 1 if rep["violations"] or "error" in rep else 0


if __name__ == "__main__":
    sys.exit(main())
