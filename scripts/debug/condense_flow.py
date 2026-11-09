#!/usr/bin/env python3
"""Condense a --mlir-print-ir-after-all dump into a diff-based flow file.

`--mlir-print-ir-after-all` reprints the ENTIRE module after every pass, so
consecutive dumps are ~99% identical and the log balloons (tens of MB). This
tool keeps only what each pass CHANGED (unified diff vs the previous dump),
collapses no-op passes to a single line, and (optionally) drops mirror
dispatches. The result shows the overall lowering flow at a fraction of the
size.

Example:
  python scripts/debug/condense_flow.py backend_allpass.log -o backend_flow.log --keep 188
"""
import argparse, difflib, re, os, sys

# matches both plain headers ("... (arg) //----- //") and module-scope headers
# ("... (arg) ('builtin.module' operation) //----- //").
HDR = re.compile(r'^// -----// IR Dump (Before|After) (\w+) \(([^)]*)\)(?: \([^)]*\))? //----- //')


def parse_records(path, match="after"):
    records, cur, lines = [], None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = HDR.match(line)
            if m:
                if cur is not None:
                    cur["body"] = lines
                    records.append(cur)
                cur = {"dir": m.group(1), "pass": m.group(2), "arg": m.group(3)}
                lines = []
            elif cur is not None:
                lines.append(line.rstrip("\n"))
        if cur is not None:
            cur["body"] = lines
            records.append(cur)
    if match == "after":
        records = [r for r in records if r["dir"] == "After"]
    # tag pass label with direction when both kept
    for r in records:
        r["label"] = (f"[{r['dir']}] " if match == "both" else "") + f"{r['pass']} ({r['arg']})"
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="input dump (from --mlir-print-ir-after-all)")
    ap.add_argument("-o", "--out", default="flow.log", help="output flow file")
    ap.add_argument("--keep", type=int, default=0,
                    help="keep only the first N records (0 = all). Use to drop mirror dispatches.")
    ap.add_argument("--range", default="", metavar="START:END",
                    help="keep only record indices [START:END) (after --match filtering)")
    ap.add_argument("--match", choices=["after", "both"], default="after",
                    help="'after' = IR Dump After only (default); 'both' = Before+After "
                         "(captures single before/after passes like lower-to-aie)")
    ap.add_argument("--cap", type=int, default=160,
                    help="max changed lines shown per pass (repeated cores/structure elided)")
    args = ap.parse_args()

    records = parse_records(args.log, match=args.match)
    if not records:
        sys.exit("no 'IR Dump' records found")
    recs = records
    if args.range:
        s, e = args.range.split(":")
        recs = recs[int(s) if s else None:int(e) if e else None]
    if args.keep:
        recs = recs[:args.keep]
    CAP = args.cap

    out = [
        f"# {os.path.basename(args.out)} — lowering flow (diff-condensed)",
        f"# source: {args.log}  |  pass마다 반복되는 동일 IR 제거, 바뀐 줄(diff)만 표시",
        f"# 변화 없는 pass는 한 줄로 축약. 큰 전환은 최대 {CAP}줄까지(초과분 생략 표기).",
        "# 기호:  '-' 제거   '+' 추가   '@@' 위치(hunk)",
        "#" + "=" * 78,
    ]
    prev, nochange = [], []

    def flush():
        if not nochange:
            return
        if len(nochange) == 1:
            out.append(f"    · {nochange[0]}  — IR 변화 없음")
        else:
            out.append(f"    · {nochange[0]} … {nochange[-1]}  ({len(nochange)}개 pass) — IR 변화 없음")
        nochange.clear()

    for i, r in enumerate(recs):
        body = r["body"]
        tag = f"#{i} {r['label']}"
        diff = [d for d in difflib.unified_diff(prev, body, n=1, lineterm="")
                if not (d.startswith("--- ") or d.startswith("+++ "))]
        changed = [d for d in diff if d and d[0] in "+-"]
        if i == 0:
            flush()
            out += ["", "┌─ " + tag + "   [진입 IR]"]
            out += ["  " + l for l in body[:CAP]]
            if len(body) > CAP:
                out.append(f"  ... (+{len(body)-CAP} more lines)")
        elif not changed:
            nochange.append(tag)
        else:
            flush()
            out += ["", "┌─ " + tag + f"   [Δ {len(changed)} lines]"]
            shown = 0
            for d in diff:
                if d.startswith("@@") or (d and d[0] in "+-"):
                    out.append("  " + d)
                    if d[0] in "+-":
                        shown += 1
                        if shown >= CAP:
                            out.append(f"  ... (+{len(changed)-shown} more changed lines; 반복 코어/구조 생략)")
                            break
        prev = body
    flush()

    open(args.out, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"wrote {args.out}")
    print(f"  in : {os.path.getsize(args.log)/1e6:.1f} MB ({len(records)} records)")
    print(f"  out: {os.path.getsize(args.out)/1e6:.3f} MB")


if __name__ == "__main__":
    main()
