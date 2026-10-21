#!/usr/bin/env python3
"""NPU bf16 벡터화 matmul 출력을 bfp16(ebs8) 양자화 흉내와 원소 단위로 비교 (세션 2026-10-21/1 챕터 04)
** v3 (2026-10-22 세션 1 챕터 04): v2 사본에 비정규수·0 부호 처리를 더한 V3.1 (아래). v2 설명:
** v2 (2026-10-22 세션 1 챕터 03): 원본 orig/bfp_strict.py 사본에 하드웨어 누산 규칙 V3을 더했다. 양자화(bfp)는 원본과 같다. **
  V3 : 블록마다 공통 기준 지수 E = max(floor(log2|acc|), eA+eB) (acc = 0이면 eA+eB; eA·eB = 이번 K 블록 A 행·B 열의 bfp 블록 지수)
       acc와 블록합 S를 둘 다 2^(E-23) 격자로 RNE 반올림한 뒤 더하고 f32 RNE. A·B 블록 중 하나가 전부 0이면 acc 그대로.
       이 반올림은 crRnd를 따르지 않는다(floor 모드 출력도 RNE로 일치).
  V3.1(챕터 04 경계 시험으로 추가):
   (1) 블록 지수 e < -126(bf16 비정규만 있는 블록)이면 그 블록을 0으로 본다 (E3·E5. 하한 -127~-122 사이는 구분 못 함)
   (2) 덧셈 결과가 0이 아니고 |acc| < 2^-126 이면 부호를 유지한 0으로 (FTZ, E2·E6·E7)
   (3) 전부 0인 블록도 MAC은 돈다: acc + (+0) → -0은 +0이 된다(IEEE). 0이 아닌 값은 그대로 (E6·E7 0의 부호)
  python3 bfp_strict.py <npu out.npy> <a.npy(f32)> <b.npy(f32)> <rnd: floor|half_even> [label]
양자화는 10/19 세션 1 챕터 04 bfp16_sim_round.py의 bfp()(renorm)와 같다:
  입력은 CPU demote(arith.truncf, RNE)로 bf16 → K 방향 8개 묶음(A는 행, B는 열) → 블록 지수 floor(log2|max|),
  가수 8비트, step = 2^(e-6), 넘치면 지수 +1 후 다시 양자화.
누산(엄격판, 이번에 추가):
  V1 : acc_f32 = f32(acc + 블록합)            (블록합 = 8개 곱의 정확한 합, float64로 정확)
  V2 : acc_f32 = f32(acc + f32(블록합))
  F64: 10/19 방식(float64 전체 곱) — 허용 오차 비교용
보고: 비트 동일 비율(V1·V2), 10/19식 허용 오차(1e-6 × max|ref|) 안 비율, 다른 원소의 위치·크기 요약."""
import sys, numpy as np

def bf16(x):
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)

RND = {"floor": np.floor, "half_even": np.round}

def bfp(x, axis, rnd, m=8):
    x = np.moveaxis(x.astype(np.float64), axis, -1); sh = x.shape
    xb = x.reshape(*sh[:-1], sh[-1] // 8, 8); mx = np.abs(xb).max(-1, keepdims=True)
    e = np.where(mx > 0, np.floor(np.log2(np.where(mx > 0, mx, 1))), 0)
    hi = 2 ** (m - 1) - 1; lo = -hi - 1
    f = RND[rnd]; step = 2.0 ** (e - (m - 2)); q = f(xb / step)
    over = ((q > hi) | (q < lo)).any(-1, keepdims=True)
    step = np.where(over, step * 2, step); q = f(xb / step)
    q = np.clip(q, lo, hi) * step
    return np.moveaxis(q.reshape(sh), -1, axis)

def bfp_exp(x, axis, rnd, m=8):
    """bfp()와 같은 규칙으로 블록 지수만 돌려준다(넘침 재정규화 포함). 전부 0인 블록은 nan."""
    x = np.moveaxis(x.astype(np.float64), axis, -1); sh = x.shape
    xb = x.reshape(*sh[:-1], sh[-1] // 8, 8); mx = np.abs(xb).max(-1, keepdims=True)
    e = np.where(mx > 0, np.floor(np.log2(np.where(mx > 0, mx, 1))), np.nan)
    hi = 2 ** (m - 1) - 1; lo = -hi - 1; q = RND[rnd](xb / 2.0 ** (e - (m - 2)))
    e = np.where(((q > hi) | (q < lo)).any(-1, keepdims=True), e + 1, e)
    return np.moveaxis(e[..., 0], -1, axis)            # A: (M, K/8), B: (K/8, N)

EMIN_BLOCK = -126   # V3.1 (1)
def acc_hw(qa, qb, eA, eB):
    """V3.1: 하드웨어 bfp16 MAC 누산 규칙(챕터 03) + 비정규·0 부호 처리(챕터 04)."""
    eA = np.where(eA < EMIN_BLOCK, np.nan, eA); eB = np.where(eB < EMIN_BLOCK, np.nan, eB)          # (1)
    qa = np.where(np.repeat(np.isnan(eA), 8, axis=1), 0.0, qa); qb = np.where(np.repeat(np.isnan(eB), 8, axis=0), 0.0, qb)
    M, K = qa.shape; N = qb.shape[1]; acc = np.zeros((M, N), np.float32)
    for t in range(K // 8):
        S = qa[:, 8*t:8*t+8] @ qb[8*t:8*t+8, :]
        eab = eA[:, t][:, None] + eB[t, :][None, :]; ok = ~np.isnan(eab)
        a = acc.astype(np.float64)
        ea = np.where(a != 0, np.floor(np.log2(np.where(a != 0, np.abs(a), 1.0))), -np.inf)
        grid = 2.0 ** (np.maximum(ea, np.where(ok, eab, 0.0)) - 23)
        new = (np.round(a / grid) * grid + np.round(S / grid) * grid).astype(np.float32)
        new = np.where((new != 0) & (np.abs(new) < 2.0 ** -126), np.copysign(np.float32(0), new), new)   # (2)
        acc = np.where(ok, new, (a + 0.0).astype(np.float32)).astype(np.float32)                         # (3)
    return acc

def main():
    out, fa, fb, rnd = sys.argv[1:5]; label = sys.argv[5] if len(sys.argv) > 5 else out
    g = np.load(out).astype(np.float32)
    a = bf16(np.load(fa)); b = bf16(np.load(fb))
    qa = bfp(a, 1, rnd); qb = bfp(b, 0, rnd)
    M, K = qa.shape; N = qb.shape[1]
    v1 = np.zeros((M, N), np.float32); v2 = np.zeros((M, N), np.float32)
    for k in range(0, K, 8):
        s = qa[:, k:k + 8] @ qb[k:k + 8, :]           # 블록합: float64로 정확
        v1 = (v1.astype(np.float64) + s).astype(np.float32)
        v2 = (v2 + s.astype(np.float32)).astype(np.float32)
    v3 = acc_hw(qa, qb, bfp_exp(a, 1, rnd), bfp_exp(b, 0, rnd))
    eq3 = (g.view(np.uint32) == v3.view(np.uint32))
    f64 = qa @ qb
    ref = a.astype(np.float64) @ b.astype(np.float64)
    tol = 1e-6 * np.abs(ref).max()
    eq1 = (g.view(np.uint32) == v1.view(np.uint32)); eq2 = (g.view(np.uint32) == v2.view(np.uint32))
    d = np.abs(g.astype(np.float64) - f64)
    print(f"{label} [{rnd}] M{M} N{N} K{K}: 비트동일 V1 {eq1.mean()*100:.4f}% ({(~eq1).sum()}개 다름) "
          f"V2 {eq2.mean()*100:.4f}% | V3.1(하드웨어 규칙) {eq3.mean()*100:.4f}% ({(~eq3).sum()}개 다름) | 허용오차내 {np.mean(d <= tol)*100:.4f}% | "
          f"NPU vs 흉내F64 max_abs {d.max():.3g} | NPU vs bf16입력정답 mean_rel {np.abs(g - ref).mean() / np.abs(ref).mean():.3g}")
    if (~eq1).any():
        idx = np.argwhere(~eq1); dd = np.abs(g.astype(np.float64) - v1)[~eq1]
        ulp = np.abs(g.view(np.int32).astype(np.int64) - v1.view(np.int32).astype(np.int64))[~eq1]
        print(f"   다른 원소: 행 {np.unique(idx[:,0]).size}/{M}개, 열 {np.unique(idx[:,1]).size}/{N}개, "
              f"차이 max {dd.max():.3g} median {np.median(dd):.3g}, ULP max {ulp.max()} median {np.median(ulp):.0f}, "
              f"앞 5개 {[tuple(int(v) for v in r) for r in idx[:5]]}")

main()
