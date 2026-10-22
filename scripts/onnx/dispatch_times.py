"""Per-dispatch times from an iree-run-module console trace.

The trace comes from the iree-run-module in build-trace/ (runtime built with
IREE_TRACING_PROVIDER=console, see scripts/build/configure.sh), which prints
every trace zone to stderr: `►` at its begin, `◄ ... t = <ns> ... @ <start ns>`
at its end, one stack per thread.

  npu dispatch  the amdxdna dispatch zone, named by the kernel-name message
                logged inside it; its time is the sum of its normal_run zones
                (submit + wait). The rest of the enclosing HAL dispatch zone is
                hwctx create/destroy, which counts as host time.
  cpu dispatch  the workgroup zones named by the export, grouped by name (each
                dispatch runs once per call); its time is first start to last
                end over the worker threads.
  total         the iree_vm_invoke zone (the function call).
  host          total minus the dispatch times.
"""

import re
from collections import defaultdict

LINE = re.compile(r"^\[([0-9A-F]+)\]\[(.{16})\] *(►|◄|\S) (.*)$")
BEGIN = re.compile(r"^(.*) \([^()]*:\d+\)$")
END = re.compile(r"^(.*?) t = (\d+)ns / .* @ (\d+)ns$")

NPU_HAL = "iree_hal_command_buffer_dispatch"
NPU_DISPATCH = "iree_hal_amdxdna_direct_command_buffer_dispatch"
NPU_RUN = "iree_hal_amdxdna_direct_command_buffer_normal_run"


def parse(path):
    """Returns ([dispatch dict in start order], total ns, hwctx ns)."""
    stacks = defaultdict(list)  # thread id -> [[zone name, kernel, run ns]]
    npu, cpu, total = [], {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE.match(line.rstrip("\n"))
            if not m:
                continue
            tid, thread, mark, rest = m.groups()
            stack = stacks[tid]
            if mark == "►":
                b = BEGIN.match(rest)
                stack.append([b[1] if b else rest, None, 0])
                continue
            if mark != "◄":
                # A message names the zone it is logged in.
                if stack:
                    stack[-1][1] = rest.strip()
                continue
            e = END.match(rest)
            if not e or not stack:
                continue
            name, kernel, run_ns = stack.pop()
            dur, start = int(e[2]), int(e[3])
            if name == NPU_RUN:
                for frame in reversed(stack):
                    if frame[0] == NPU_DISPATCH:
                        frame[2] += dur
                        break
            elif name == NPU_DISPATCH:
                if stack and stack[-1][0] == NPU_HAL:
                    stack[-1][1], stack[-1][2] = kernel, run_ns
            elif name == NPU_HAL and kernel is not None:
                npu.append({"name": kernel, "target": "npu", "start": start,
                            "ns": run_ns, "hwctx_ns": dur - run_ns})
            elif name == "iree_vm_invoke":
                total = dur
            elif thread.startswith("iree-worker") and not name.startswith("iree_"):
                span = cpu.setdefault(name, [start, start + dur])
                span[0], span[1] = min(span[0], start), max(span[1], start + dur)
    if total is None:
        raise ValueError(f"no iree_vm_invoke zone in {path}; not a console trace?")
    dispatches = npu + [{"name": k, "target": "cpu", "start": s, "ns": e - s,
                         "hwctx_ns": 0} for k, (s, e) in cpu.items()]
    dispatches.sort(key=lambda d: d["start"])
    return dispatches, total, sum(d["hwctx_ns"] for d in dispatches)


def format_report(dispatches, total, hwctx):
    """Returns the per-dispatch lines and the per-target summary."""
    width = max((len(d["name"]) for d in dispatches), default=0)
    lines = [f"#{i:<3} {d['name']:<{width}} -> {d['target']}  {d['ns'] / 1e6:9.3f} ms"
             for i, d in enumerate(dispatches)]
    lines.append("---")
    busy = 0
    for target in sorted({d["target"] for d in dispatches}, key=["npu", "cpu"].index):
        sel = [d for d in dispatches if d["target"] == target]
        ns = sum(d["ns"] for d in sel)
        busy += ns
        lines.append(f"{target:<5} {len(sel):3d} dispatches  {ns / 1e9:8.3f} s")
    lines.append(f"host                  {(total - busy) / 1e9:8.3f} s"
                 f"  (hwctx create/destroy {hwctx / 1e9:.3f} s)")
    lines.append(f"total                 {total / 1e9:8.3f} s")
    return "\n".join(lines)
