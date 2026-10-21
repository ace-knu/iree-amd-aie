#!/usr/bin/env python3
"""v9 회귀 판정기. 첫 줄에 "OK …" 또는 "NG …"를 찍고, 종료 코드 0(통과)/1(실패)을 돌려준다. /opt/venv의 numpy로 실행.
  judge.py bfp      <out.npy> <in_dir> <floor|half_even> [old]   V1·V2: V3.1 흉내(tools/bfp16_emulator)와 원소 100% 비트 일치.
                                                                 old = 옛 누산 모델(블록 합을 f32 RNE로)로 판정(검출력 확인용)
  judge.py bfp_neg  <out.npy> <in_dir> <floor|half_even> <max%>  V2: 다른 반올림 흉내와의 비트 일치율이 max% 미만이어야 통과
  judge.py int      <out.npy> <in_dir>                           V3: 정수 정답 a@b(int64)와 원소 전부 같음
  judge.py ref      <식> <in_dir> <rel> <out1.npy> [out2.npy]    V4~V6: float64 기준 식과 출력 전체 비교. 식: chain2, mlp_reduce, fanout3, fanout3_cpu, chain3
                    (환경 변수 REG_JUDGE_REFDIR가 있으면 기준을 그 폴더 입력으로 계산 — 검출력 확인용)
  judge.py allclose <rel> <n> <out1..outn> <ref1..refn>         V0: 장치 출력과 CPU 기준 출력(npy)을 원소별로
  judge.py same     <a.npy> <b.npy> …                            V5: 모두 비트 동일
  judge.py order    <stamped_stderr> <n2g|g2n>                   V5: 로그 순서(10/19 analyze.py와 같은 로그 규칙)
  judge.py sha      <out.npy> <expected_sha256>                  F1~F3
상대 오차 rel = max|out − ref| / max|ref| (NaN이 있으면 실패)."""
import os, sys, re, hashlib, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMU_DIR = os.environ.get("REG_EMU_DIR", os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "bfp16_emulator")))


def emu():
    sys.path.insert(0, EMU_DIR)
    import bfp_strict_v3 as m   # 흉내 V3.1 (복사하지 않고 import, 2026-10-22 세션 2에 __main__ 보호 추가)
    return m


def done(ok, msg):
    print(("OK " if ok else "NG ") + msg)
    sys.exit(0 if ok else 1)


def bfp_expect(m, a, b, rnd, model):
    a, b = m.bf16(a), m.bf16(b)
    qa, qb = m.bfp(a, 1, rnd), m.bfp(b, 0, rnd)
    if model == "old":
        acc = np.zeros((a.shape[0], b.shape[1]), np.float32)
        for k in range(0, a.shape[1], 8): acc = (acc.astype(np.float64) + qa[:, k:k + 8] @ qb[k:k + 8, :]).astype(np.float32)
        return acc
    return m.acc_hw(qa, qb, m.bfp_exp(a, 1, rnd), m.bfp_exp(b, 0, rnd))


def bitpct(g, e):
    eq = g.astype(np.float32).view(np.uint32) == e.astype(np.float32).view(np.uint32)
    return eq.mean() * 100, int((~eq).sum()), np.argwhere(~eq)[:3].tolist()


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)


def ref_formula(name, d):
    a, b, c = (np.load(f"{d}/{n}.npy").astype(np.float64) for n in ("a", "b", "c"))
    return {"chain2": [(a @ b) @ c], "mlp_reduce": [softmax(a @ b) @ c], "fanout3": [(a @ b) @ c, softmax(a @ b)],
            "fanout3_cpu": [softmax(a) @ b, softmax(a) @ c], "chain3": [softmax((a @ b) @ c)]}[name]


def relerr(o, r):
    o = o.astype(np.float64); r = r.astype(np.float64)
    if o.shape != r.shape: return float("inf"), True
    return float(np.abs(o - r).max() / max(np.abs(r).max(), 1e-30)), bool(np.isnan(o).any())


def main():
    mode = sys.argv[1]; A = sys.argv[2:]
    if mode in ("bfp", "bfp_neg"):
        m = emu(); g = np.load(A[0]); a = np.load(f"{A[1]}/a.npy"); b = np.load(f"{A[1]}/b.npy"); rnd = A[2]
        if mode == "bfp":
            model = A[3] if len(A) > 3 else "v31"
            p, n, w = bitpct(g, bfp_expect(m, a, b, rnd, model))
            p1, n1, _ = bitpct(g, bfp_expect(m, a, b, rnd, "old")) if model != "old" else (p, n, None)
            done(n == 0, f"bfp({rnd},{'V3.1' if model != 'old' else '옛 누산'}) 비트일치 {p:.4f}% ({n}개 다름{', 처음 ' + str(w) if n else ''}) | 참고 옛 누산 {p1:.4f}%")
        mx = float(A[3]); p, n, _ = bitpct(g, bfp_expect(m, a, b, rnd, "v31"))
        done(p < mx, f"bfp_neg({rnd}) 비트일치 {p:.4f}% (기준 < {mx}%)")
    if mode == "int":
        g = np.load(A[0]).astype(np.int64); a = np.load(f"{A[1]}/a.npy").astype(np.int64); b = np.load(f"{A[1]}/b.npy").astype(np.int64)
        ok = g.shape == (a.shape[0], b.shape[1]) and bool((g == a @ b).all())
        done(ok, f"int 정수 정답과 {'같음' if ok else '다름'} (다른 원소 {int((g != a @ b).sum()) if g.shape == (a.shape[0], b.shape[1]) else '모양 다름'}) | sha256 {hashlib.sha256(open(A[0], 'rb').read()).hexdigest()[:16]}")
    if mode == "ref":
        name, d, rel, outs = A[0], A[1], float(A[2]), A[3:]
        refs = ref_formula(name, os.environ.get("REG_JUDGE_REFDIR") or d)
        if len(refs) != len(outs): done(False, f"ref({name}) 출력 수 {len(outs)} ≠ 기준 수 {len(refs)}")
        errs = [relerr(np.load(o), r) for o, r in zip(outs, refs)]
        ok = all(e <= rel and not nan for e, nan in errs)
        done(ok, f"ref({name}) rel " + ", ".join(f"{e:.2e}{' NaN' if nan else ''}" for e, nan in errs) + f" (기준 ≤ {rel:g})")
    if mode == "allclose":
        rel, n = float(A[0]), int(A[1]); outs, refs = A[2:2 + n], A[2 + n:2 + 2 * n]
        errs = [relerr(np.load(o), np.load(r)) for o, r in zip(outs, refs)]
        ok = len(outs) == n and len(refs) == n and all(e <= rel and not nan for e, nan in errs)
        done(ok, "allclose rel " + ", ".join(f"{e:.2e}{' NaN' if nan else ''}" for e, nan in errs) + f" (기준 ≤ {rel:g})")
    if mode == "same":
        hs = [hashlib.sha256(np.load(f).tobytes()).hexdigest() for f in A]
        done(len(set(hs)) == 1, f"same {len(A)}개 비트 {'동일' if len(set(hs)) == 1 else '다름'} ({hs[0][:12]})")
    if mode == "order":
        st, kind = A[0], A[1]; ev = []
        for line in open(st):
            mm = re.match(r"^([\d.]+)\t(.*)$", line.rstrip("\n"))
            if mm: ev.append((float(mm.group(1)), mm.group(2)))
        npu_begin = [t for t, l in ev if l.startswith("[T] submit begin")]
        npu_done = [t for t, l in ev if re.search(r"\[T\] submit '[^']+' issue=", l)]
        gsub = [t for t, l in ev if re.search(r"\[BR\] vk submit cmdbufs=[1-9]", l)]
        gdone = [t for t, l in ev if re.search(r"\[BR\] vk complete sem=\S+ value=[1-9]", l)]
        if kind == "n2g":
            if not npu_done or not gsub: done(False, f"order n2g 로그 없음 (npu 완료 {len(npu_done)}, vk submit {len(gsub)})")
            done(gsub[0] > npu_done[0], f"order n2g GPU 제출 {gsub[0]:.1f} ms {'>' if gsub[0] > npu_done[0] else '≤'} NPU 완료 {npu_done[0]:.1f} ms")
        if not npu_begin or not gdone: done(False, f"order g2n 로그 없음 (npu 시작 {len(npu_begin)}, vk complete {len(gdone)})")
        done(npu_begin[0] > gdone[0], f"order g2n NPU 시작 {npu_begin[0]:.1f} ms {'>' if npu_begin[0] > gdone[0] else '≤'} GPU 완료 {gdone[0]:.1f} ms")
    if mode == "sha":
        h = hashlib.sha256(open(A[0], "rb").read()).hexdigest()
        done(h == A[1], f"sha {h[:16]}… {'=' if h == A[1] else '≠'} 기준 {A[1][:16]}…")
    sys.exit(f"알 수 없는 판정: {mode}")


if __name__ == "__main__":
    main()
