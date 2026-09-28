---
date: 2026-09-28
topic: [team-share]
status: reference
summary: "bert-onnx 받는 법: third_party/iree 가 ace-knu/iree 의 bert-onnx 브랜치를 가리키도록 정정(IREE 패치 불필요), torch-mlir 패치 1개만 patches/third_party/, LayerNorm 에 필요한 것"
---
# `bert-onnx` 받는 법 — 서브모듈 정정과 남은 패치 하나 (2026-09-28)

`docs/2026-09-11_bert_onnx_changes_for_teammates.md` 의 후속입니다. **"LayerNorm 이 아예 안 된다"**
는 제보를 받고 푸쉬된 버전을 확인했습니다. 원인은 두 가지였고 둘 다 제 쪽 실수입니다.

## 1. 무엇이 잘못돼 있었나

### ① 서브모듈 포인터가 받을 수 없는 커밋을 가리켰습니다

9/18 커밋 `aefbb6d` 가 `third_party/iree` 를 `c73928f` → **`8d2091b`** 로 올려 두었는데,
`8d2091b` 는 `ace-knu/iree` 에 **없었습니다**(제 로컬에만 있음). 9/11 에 `5cea157` 로 고쳤던 것과
같은 실수입니다. 그래서 9/18 이후 `git submodule update` 가 받을 수 없는 ref 로 실패하고,
서브모듈은 이전 상태로 남았습니다.

### ② LayerNorm 에 필요한 IREE 수정을 드린 적이 없었습니다

LayerNorm 을 NPU 로 올리는 경로는 IREE 본체의 버그 수정 하나가 필요합니다.
`iree_linalg_ext.custom_op` 이 심볼 차원 범위를 잘못된 인덱스로 읽어서, 루프 1 개·심볼 2 개인
LayerNorm 을 raise 하는 순간 **`iree-compile` 이 assert 로 죽습니다**(release 빌드면 범위 밖
읽기). 심볼 개수와 루프 개수가 같을 때만 우연히 맞아서 다른 모델에서는 드러나지 않았습니다.
이 수정과 head-split broadcast 수정은 문서에 경로만 있었고, 그 경로는 git 에 안 올라가는
위치라 받으실 방법이 없었습니다.

## 2. 무엇을 바꿨나 — IREE 는 이제 패치가 아니라 서브모듈 커밋입니다

지금까지 IREE 쪽 수정을 패치 파일로 드린 이유는 "그 커밋이 공유 리모트에 없어서"였습니다.
이번에 **`ace-knu/iree` 에 `bert-onnx` 브랜치를 만들어 올렸고**, 서브모듈 포인터가 그 끝을
가리킵니다:

```
ace-knu/iree  bert-onnx:  c73928f (amd-aie-flow-hook) ─ 43018fe ─ 0db1789 ─ 4067842
```

| 커밋 | 내용 | 예전 패치 |
|---|---|---|
| `43018fe` | `DetachElementwiseFromNamedOps` 가 broadcast 로 접힌 accumulator init 을 되돌리지 않게 | 0002 |
| `0db1789` | 순수 broadcast 를 contraction 소비자 안으로 복제 | 0003 |
| **`4067842`** | **custom_op 심볼 범위 인덱스 — LayerNorm crash 수정** | 0004 |

`amd-aie-flow-hook` 은 건드리지 않았습니다. 위 세 커밋은 우리 브랜치에만 있고, 그 브랜치는
rebase/force-push 하지 않습니다(예전 포인터가 깨지므로).

**남은 패치는 torch-mlir 하나**입니다. IREE 가 upstream `iree-org/torch-mlir` 을 서브모듈로 쓰는데
거기는 푸쉬할 수 없어서입니다. 8/25 이후 바뀐 적 없는 그 패치이고, 이제 저장소 안
`patches/third_party/` 에 있습니다.

## 3. 받는 법

```bash
git fetch origin && git checkout bert-onnx && git pull
git submodule update --init --recursive     # third_party/iree → 4067842 (IREE 수정 포함)
./patches/third_party/apply.sh               # torch-mlir 패치 1 개 (두 번 돌려도 안전)
# 그다음 IREE 재빌드
```

⚠️ **예전에 0002/0003/0004 를 `third_party/iree` 에 손으로 적용하셨다면** 먼저 되돌리세요 —
`git -C third_party/iree status` 가 깨끗해야 `submodule update` 가 새 커밋으로 넘어갑니다
(`git apply` 로 붙이셨으면 `git -C third_party/iree stash`, `git am` 으로 붙이셨으면 그 커밋들은
그대로 두셔도 `submodule update` 가 새 커밋으로 체크아웃합니다).
0001(torch-mlir)은 이미 적용돼 있으면 `apply.sh` 가 건너뜁니다.

이 조합(`4067842` + torch-mlir 패치)은 제가 검증에 쓴 트리와 같은 코드입니다(대조 확인).

## 4. LayerNorm 이 NPU 로 가려면 (위 외에)

- 컴파일 플래그: `--iree-amdaie-enable-ukernels=softmax,layernorm`
  (+ 평소 쓰는 `--iree-amd-aie-enable-chess-for-ukernel=false`,
  `--iree-amd-aie-peano-install-dir=...`, `--iree-flow-enable-executable-deduplication=false`)
- 스택 크기는 컴파일러가 LayerNorm 코어에 자동으로 4096 을 줍니다.
- **모델 조건**: LayerNorm 뒤에 `QuantizeLinear` 가 있고, residual 입력이 순수
  `DequantizeLinear(i8)` 여야 NPU 로 올라갑니다. bias 가 K 축에 접히지 않은 projection 뒤의
  LayerNorm 은 residual 이 `dequant → bias Add → residual Add` 가 되어 **CPU 에 남습니다**
  (crash 가 아니라 CPU 배치). 12 층에서는 24 개 중 15 개가 NPU, 9 개가 CPU 입니다.

## 5. 아직 이 브랜치에 없는 것

- ~~11 층 이상에서의 hang 수정~~ → **같은 날 추가됨**(`798bbc4`, `d4aadc9`). 12 층 인코더를 통째로
  돌릴 때 입력에 따라 layer 10 QK^T 에서 나던 `ert state 8` 타임아웃의 수정입니다. 원인은 그 층의
  재양자화 스케일에서 정수 변환이 거부되어 float 재양자화가 코어에 남고, 소프트 플로트 코어가 교착한
  것이었습니다. 이제 정확히 같은 결과를 내는 정수 표현을 더 넓게 찾고(`798bbc4`), 기본값으로는
  실행 안전 조건만 보고 정수화합니다(`d4aadc9`, 12 층 출력 바이트 동일 확인). 이전 동작은
  `--iree-amdaie-force-integer-requantization=false`.
- 모델 준비 스크립트(i8 경계 추출, 범용 K축 bias 접기, head split)는 정리 전이라 브랜치에 없습니다.
