#!/usr/bin/env python3
"""v9 회귀 입력 생성기. 입력·기준을 실행 때 시드로 만든다(저장소에 npy를 두지 않음). /opt/venv의 numpy로 실행.
  gen_inputs.py v0   <out_dir> <n>                 V0: 128x128 f32 입력 n개 v0_in<i>.npy (default_rng(7000), N(0,1)*0.3)
  gen_inputs.py bf16 <out_dir> <M> <N> <K> <rand|relu> [seed=3]   V1·V2: a.npy(MxK) b.npy(KxN) f32 (10/21 rnd_test.sh·10/19 cmp3.sh와 같은 생성기)
  gen_inputs.py i8   <out_dir> <M> <N> <K> [seed=7]                V3: a.npy b.npy int8 (10/21 rnd_test.sh와 같음)
  gen_inputs.py r128 <out_dir> <seed>                              V4~V6: a.npy b.npy c.npy 128x128 f32 (default_rng(1000+seed), N(0,1)*0.3)
  gen_inputs.py sig  <out_dir> <mlir> [seed=11]                    V7: 함수 인자 tensor<AxBxf32> 순서대로 in<k>.npy (default_rng(seed), U(-1,1)*0.25; 10/22 세션 6 run_pd.sh와 같음)"""
import os, sys, numpy as np
kind, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); f32 = np.float32
if kind == "v0":
    r = np.random.default_rng(7000)
    for i in range(int(sys.argv[3])): np.save(f"{out}/v0_in{i}.npy", (r.standard_normal((128, 128)) * 0.3).astype(f32))
elif kind == "sig":
    import re
    src = open(sys.argv[3]).read(); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 11
    r = np.random.default_rng(seed); sig = re.search(r"func\.func @\w+\((.*?)\) ->", src).group(1)
    for k, (x, y) in enumerate(re.findall(r"tensor<(\d+)x(\d+)xf32>", sig)):
        np.save(f"{out}/in{k}.npy", (r.uniform(-1, 1, (int(x), int(y))) * 0.25).astype(f32))
elif kind == "bf16":
    M, N, K = map(int, sys.argv[3:6]); k = sys.argv[6]; seed = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    rng = np.random.default_rng(seed)
    if k == "rand":
        a = (rng.standard_normal((M, K)) * 0.5).astype(f32); b = (rng.standard_normal((K, N)) * 0.2).astype(f32)
    else:
        z = rng.standard_normal((M, K))
        a = np.where(z > 0.3, np.abs(z) * 10.0 ** rng.uniform(-2, 1, (M, K)), 0.0).astype(f32)
        b = (rng.standard_normal((K, N)) * 0.05).astype(f32)
    np.save(f"{out}/a.npy", a); np.save(f"{out}/b.npy", b)
elif kind == "i8":
    M, N, K = map(int, sys.argv[3:6]); seed = int(sys.argv[6]) if len(sys.argv) > 6 else 7
    rng = np.random.default_rng(seed)
    np.save(f"{out}/a.npy", rng.integers(-128, 128, (M, K), dtype=np.int8)); np.save(f"{out}/b.npy", rng.integers(-128, 128, (K, N), dtype=np.int8))
elif kind == "r128":
    r = np.random.default_rng(1000 + int(sys.argv[3]))
    for n in ("a", "b", "c"): np.save(f"{out}/{n}.npy", (r.standard_normal((128, 128)) * 0.3).astype(f32))
else:
    sys.exit(f"알 수 없는 종류: {kind}")
