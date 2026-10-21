# bfp16_emulator — AIE2P bfp16 MAC 하드웨어 흉내 (V3.1)

AIE2P(npu4) 벡터화 bf16 matmul의 NPU 출력을 **비트 단위로** 재현하는 numpy 흉내다. 커널 정확도 기준은 "흉내와 원소 100% 비트 일치"이고, 정규분포·ReLU 흉내·실제 모델 데이터에 모두 쓸 수 있다. 규칙과 근거는 명세 `docs/specs/S12_AIE2P_bfp16_MAC_하드웨어모델.md`에 있다.

- 대상:
  - 규민 aievec 하강 경로(`--iree-amdaie-enable-vectorization-passes=true`)의 bf16 matmul
  - Peano bfp16 ukernel 경로(`--iree-amdaie-enable-ukernels=matmul --iree-amd-aie-enable-chess-for-ukernel=0`)
  - 두 경로의 하드웨어 출력은 비트 단위로 같다. 둘 다 `BFP576.BFP576.ACC2048.mac.conf` conf 780을 쓴다.
- 대상 아님: AIE2(npu1)의 벡터화 bf16은 bf16 MAC(`bf.mac16.conf`)이라 이 규칙이 아니다(미확인). 스칼라(벡터화 끔) 경로도 대상이 아니다.
- 필요한 것: python3, numpy

## 사용법
```bash
python3 bfp_strict_v3.py <NPU 출력.npy> <A.npy> <B.npy> <floor|half_even> [이름]
```
- `A.npy`(M×K)와 `B.npy`(K×N)는 **f32** 입력이다. 흉내 안에서 bf16 RNE로 바꾼다(팀 demote `--iree-amdaie-demote-contraction-inputs-to-bf16`와 같음). 이미 bf16 값이면 그대로다.
- 반올림 인자는 코어 crRnd 설정에 맞춘다:
  - 기본(`--iree-amdaie-core-rounding-mode` 생략 = AIE2P conv_even) → `half_even`
  - `--iree-amdaie-core-rounding-mode=floor` → `floor`
- 출력 한 줄에서 **`V3.1(하드웨어 규칙) 100.0000% (0개 다름)`**이면 비트 일치다. 그 밖의 값은 참고용이다:
  - `V1`·`V2`: 블록 합을 f32 RNE로 누산하는 옛 모델. ReLU류 입력에서 틀린다.
  - 허용오차: 1e-6·max|정답|
  - 정답 대비 mean_rel
- 다르면 둘째 줄에 다른 원소의 행·열 수, 크기, 앞 5개 위치를 찍는다(V1 기준).
- 제약:
  - K는 8의 배수여야 한다. 아니면 A의 열과 B의 행을 0으로 채워 8의 배수로 만든다(하드웨어 pack 패딩과 같음, VGG16 conv1_1 K=27로 확인).
  - B가 전치형(N×K)이면 `.T`로 바꿔 넣는다.

## 함수로 쓰기
```python
ns = {}; exec(open("bfp_strict_v3.py").read().split("\ndef main")[0], ns)
bf16, bfp, bfp_exp, acc_hw = ns["bf16"], ns["bfp"], ns["bfp_exp"], ns["acc_hw"]
a, b = bf16(A_f32), bf16(B_f32)                       # (M,K), (K,N)
out = acc_hw(bfp(a, 1, "half_even"), bfp(b, 0, "half_even"), bfp_exp(a, 1, "half_even"), bfp_exp(b, 0, "half_even"))   # (M,N) f32
```
- `bfp(x, axis, rnd)`: K 방향 8개 블록 bfp16 양자화(값)
- `bfp_exp(x, axis, rnd)`: 블록 지수
- `acc_hw(qa, qb, eA, eB)`: 하드웨어 누산(V3.1)

## 규칙 요약 (자세한 것은 S12)
```
양자화: bf16 → K 방향 8개 블록(A는 행, B는 열), 블록 지수 e = floor(log2 max|x|), 가수 8비트 2의 보수,
        반올림은 crRnd, 가수가 -128~127을 넘으면 e+1로 다시 양자화
(1) 블록 지수 e < -126 (bf16 비정규만 있는 블록) → 그 블록을 0으로
K 블록마다:  S = Σ qA·qB (정확)
            E = max(floor(log2|acc|), eA+eB)                      (acc = 0이면 eA+eB)
            new = f32_RNE( RNE(acc, 2^(E-23)) + RNE(S, 2^(E-23)) )   ← 정렬 반올림은 crRnd와 무관하게 RNE
(2) new ≠ 0 이고 |new| < 2^-126 → 부호를 유지한 0
(3) A·B 블록 중 하나가 전부 0 → acc = acc + (+0)
```

## 이력
- 파일 내용은 2026-10-22 세션(로컬 기록 `docs/2026-10-22/1_반올림_검증과_미해결조사/04_…/bfp_strict_v3.py`)에서 검증한 파일과 **바이트 동일**하다. sha256 `9b2d3337dcd0baa0cf64894b19a36c76722a20badec792ad56d45f361ea53941`.
- 파일 머리 주석의 "챕터"·"E2·E6" 같은 이름은 그 세션의 시험 이름이다.
- 판 이력:
  - V1: 2026-10-21, f32 RNE 누산
  - V3: 2026-10-22 챕터 03, 공통 지수 정렬
  - V3.1: 2026-10-22 챕터 04, 비정규·0 부호
