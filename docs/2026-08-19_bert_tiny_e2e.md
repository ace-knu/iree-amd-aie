# BERT-tiny end-to-end: 진행 과정 기록 (2026-08-19)

VGG16처럼 "matmul은 NPU, 나머지는 CPU"로 BERT를 붙여본 기록. 실제로 어떤 IR을 거쳐서
컴파일되는지, 왜 중간에 크래시가 났는지, 어떻게 고쳤는지를 순서대로 정리한다. 결과물
자체(레시피, 플래그 표)는 `models/bert_tiny/README.md`에 있고, 이 문서는 "과정"에
집중한다.

## 0. 전체 흐름 한눈에 보기

```
prajjwal1/bert-tiny (HF)
    │  torch.onnx.export (dynamo=False, attn_implementation="eager")
    ▼
bert_tiny.onnx  ── 이 문서 §1
    │  import_onnx
    ▼
torch/onnx 방언 MLIR  ── §2
    │  ConvertTorchToLinalg
    ▼
linalg 방언 (linalg.batch_matmul 등장)  ── §3
    │  AMDAIEAssignDeviceAffinities + FormDispatchRegions
    ▼
flow.dispatch (stream.affinity = @npu / @cpu로 쪼개짐)  ── §4
    │  AMDAIELoweringStrategy + AMDAIETileAndFuse + ...
    ▼
amdaie 방언 (tile/core/DMA 배정)  ── §5  ★ 여기서 크래시 발생 ★
    │  (원인 규명 후 seq_len 16→32로 회피)
    ▼
aie.device + aiex.runtime_sequence → .vmfb
    │  iree-run-module (실제 npu4 하드웨어)
    ▼
last_hidden_state, torch 대비 corr=0.99998  ── §7
```

## 1. 모델 준비: 왜 tokenizer 없이, 왜 eager로 export했나

`prajjwal1/bert-tiny`는 2020년경 올라온 모델이라 fast-tokenizer 파일(`tokenizer.json`)이
없다. 최신 `transformers`(5.15.0)는 legacy `vocab.txt`만 있는 저장소를 fast tokenizer로
자동 변환하는 경로가 막혀 있어서(`sentencepiece`를 깔아도 동일 에러), tokenizer 로딩
자체가 실패한다. 그런데 이 실험은 "언어 이해"가 아니라 "컴퓨트 그래프가 NPU/CPU로 잘
쪼개져서 도는가"만 보는 것이므로, tokenizer를 아예 빼고 `[0, vocab_size)` 범위의 랜덤
정수를 `input_ids`로 바로 사용했다 (`models/bert_tiny/export_bert_tiny.py`).

`attn_implementation="eager"`를 지정한 이유는 별개 문제다: HF의 기본 attention
구현(`sdpa`)은 attention mask를 준비하는 공용 유틸(`masking_utils.py`)이 `Equal`,
`Where`, `GatherElements`, `IsNaN`, `ConstantOfShape` 같은 op을 그래프에 추가로
끼워 넣는다 (mask가 전부 1이어도 마찬가지). `eager`는 attention을 `MatMul → Mul(스케일)
→ Softmax → MatMul`이라는 단순한 형태로 남겨서, import_onnx/torch-mlir가 이해하는
표준적인 op만 남기기 위한 선택이었다. **batched matmul이 NPU에서 도는 것 자체는
`eager`와 무관하다** — 이건 §3에서 확인.

## 2. import_onnx: torch/onnx 방언

가장 작은 반례(`two_bmm.onnx`, batched matmul 2개만 있는 그래프)로 보면 이 단계의
결과가 뭔지 명확하다:

```mlir
module {
  func.func @two_bmm(%arg0: !torch.vtensor<[2,16,64],f32>, %arg1: !torch.vtensor<[2,64,16],f32>,
                      %arg2: !torch.vtensor<[2,16,64],f32>) -> !torch.vtensor<[2,16,64],f32> {
    %0 = torch.operator "onnx.MatMul"(%arg0, %arg1)
         : (!torch.vtensor<[2,16,64],f32>, !torch.vtensor<[2,64,16],f32>) -> !torch.vtensor<[2,16,16],f32>
    %1 = torch.operator "onnx.MatMul"(%0, %arg2)
         : (!torch.vtensor<[2,16,16],f32>, !torch.vtensor<[2,16,64],f32>) -> !torch.vtensor<[2,16,64],f32>
    return %1 : !torch.vtensor<[2,16,64],f32>
  }
}
```

`onnx.MatMul`은 아직 이 시점엔 그냥 "torch dialect가 감싸고 있는 opaque한 ONNX
operator" 취급이다. rank(2D냐 3D batched냐)는 타입에 이미 박혀 있지만, 아직 아무 pass도
이걸 "이건 NPU로 보낼 contraction이다"라고 판단하지 않은 상태.

## 3. torch → linalg: `linalg.batch_matmul`이 등장하는 지점

`ConvertTorchToLinalg` pass를 지나면 각 `onnx.MatMul`이 rank에 맞는 linalg named op으로
내려간다. 2D면 `linalg.matmul`, 3D(batch 차원 포함)면 `linalg.batch_matmul`:

```mlir
%7 = linalg.batch_matmul ins(%2, %1 : tensor<2x16x64xf32>, tensor<2x64x16xf32>)
                         outs(%6 : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
%12 = linalg.batch_matmul ins(%cast, %0 : tensor<2x16x16xf32>, tensor<2x16x64xf32>)
                          outs(%11 : tensor<2x16x64xf32>) -> tensor<2x16x64xf32>
```

**여기가 지난 대화에서 나온 질문("batch matmul이 왜 되냐")의 답이 실제로 확인되는
지점이다.** AMD-AIE의 device-affinity 배치 로직(`AMDAIEAssignDeviceAffinities.cpp`)은
`linalg::isaContractionOpInterface`라는 MLIR 구조적 인터페이스로 판단하는데, 이 인터페이스는
indexing map 기반이라 rank에 무관하다 — `linalg.matmul`이든 `linalg.batch_matmul`이든
같은 검사를 통과한다. 즉 "2D matmul 지원 코드가 batch 차원을 하드코딩으로 막아두지 않았다"는
것 뿐이고, batched matmul을 위한 별도 지원 코드가 있는 게 아니다.

## 4. Device affinity + dispatch 분리: `stream.affinity`로 눈에 보이는 NPU/CPU 분리

`AMDAIEAssignDeviceAffinities`와 dispatch 형성 pass들을 지나면, 그래프가
`flow.executable` 여러 개로 쪼개지고 각 dispatch 호출에 `stream.affinity`가 붙는다.
같은 `two_bmm.onnx`의 최종 형태 (bf16 변환용 elementwise dispatch는 CPU로, 두 개의
`linalg.batch_matmul`은 NPU로):

```mlir
module attributes {stream.affinity.default = #hal.device.affinity<@npu>,
    stream.topology = #hal.device.topology<links = [(@npu -> @cpu = {}), (@cpu -> @npu = {transparent_access = true})]>} {
  ...
  %4 = flow.dispatch @two_bmm$async_dispatch_0::@..._elementwise_2048_f32xbf16(%3)
       {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<2048xf32>) -> tensor<2048xbf16>
  ...
  %9 = flow.dispatch @two_bmm$async_dispatch_2::@..._batch_matmul_2x16x16x64_bf16xbf16xf32(%5, %8)
       {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x16x64xbf16>, tensor<2x64x16xbf16>) -> tensor<2x16x16xf32>
  ...
  %16 = flow.dispatch @two_bmm$async_dispatch_5::@..._batch_matmul_2x16x64x16_bf16xbf16xf32(%12, %15)
       {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x16x16xbf16>, tensor<2x16x64xbf16>) -> tensor<2x16x64xf32>
}
```

f32→bf16 변환(`arith.truncf`, dispatch_0/1/3/4)은 `@cpu`, 두 batched matmul(dispatch_2,
dispatch_5)은 `@npu`. 이게 실제 IR 레벨에서 본 "matmul만 NPU" 배치의 증거다. 새로 만든
BERT 전용 코드는 없다 — VGG16과 완전히 같은 pass가 그대로 적용된 것.

## 5. AMD-AIE 코드젠: 여기서 크래시가 난다

`AMDAIELoweringStrategy` pass가 각 batched matmul에 타일링/패킹 설정을 붙인다. **성공한
dispatch_2**(QK^T 모양, `M=16,K=64,N=16`)와 **크래시한 dispatch_5**(Attn@V 모양,
`M=16,K=16,N=64`)를 나란히 보면:

```mlir
// dispatch_2 (M=16, K=64, N=16) — 성공
%7 = linalg.batch_matmul {
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 16, 16, 0], ...]>,
  packing_config = #amdaie.packing_config<packing_config = [
    {packedSizes = [0, 16, 16, 32], ...},   // M=16, N=16 -> N/16 = 1개 N-타일
    {packedSizes = [0, 0, 0, 0, 8, 8, 8], ...}]>
} ins(%3, %4 : tensor<2x16x64xbf16>, tensor<2x64x16xbf16>) outs(%6 : tensor<2x16x16xf32>) -> ...

// dispatch_5 (M=16, K=16, N=64) — 크래시
%7 = linalg.batch_matmul {
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 16, 64, 0], ...]>,
  packing_config = #amdaie.packing_config<packing_config = [
    {packedSizes = [0, 16, 8, 16], ...},    // M=16, N=64 -> N/8 = 8개 N-타일
    {packedSizes = [0, 0, 0, 0, 8, 8, 8], ...}]>
} ins(%3, %4 : tensor<2x16x16xbf16>, tensor<2x16x64xbf16>) outs(%6 : tensor<2x16x64xf32>) -> ...
```

N=16일 땐 N-타일이 1개, N=64일 땐 N-타일이 8개 필요하다는 게 packing_config에 그대로
드러난다. 이후 `AMDAIEFlattenLogicalObjectFifo` pass 직후 실제로 물리 타일에 배정되는
`amdaie.tile(col, row)` 값을 비교하면 원인이 그대로 보인다:

```mlir
// dispatch_2 (성공) — row 0,1,2 만 사용
%tile_0_1 = amdaie.tile(%c0, %c1)
%tile_0_0 = amdaie.tile(%c0, %c0)
%tile_0_2 = amdaie.tile(%c0, %c2)

// dispatch_5 (크래시) — row 0..9, 총 10개 row를 요구
%tile_0_1 = amdaie.tile(%c0, %c1)
%tile_0_0 = amdaie.tile(%c0, %c0)
%tile_0_3 = amdaie.tile(%c0, %c3)
%tile_0_4 = amdaie.tile(%c0, %c4)
%tile_0_5 = amdaie.tile(%c0, %c5)
%tile_0_6 = amdaie.tile(%c0, %c6)
%tile_0_7 = amdaie.tile(%c0, %c7)
%tile_0_8 = amdaie.tile(%c0, %c8)
%tile_0_9 = amdaie.tile(%c0, %c9)   // <- npu4는 #hal.executable.target{num_rows = 4}
%tile_0_2 = amdaie.tile(%c0, %c2)
```

`#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8, num_rows = 4, ...}>` —
npu4는 컬럼당 row가 4개뿐인데, dispatch_5는 8개의 N-타일 결과를 컬럼 0 안에서 row
2~9로 나열하려고 한다. row 6부터는 물리적으로 존재하지 않는 좌표라 `getTileType(col=0,
row=9)`가 "Cannot find Tile Type"으로 죽는다. 즉 **N-타일 개수가 물리 row 수(4)를
넘으면 다음 컬럼으로 넘어가야 하는데, 지금 코드는 같은 컬럼 안에서 row만 계속
늘린다** — 이게 실제 버그의 정체다.

## 6. 크래시를 어떻게 좁혀왔나 (증상 → 최소 재현)

이 과정 자체가 흥미로워서 순서대로 남긴다.

1. BERT 전체 컴파일 → `AMDAIEDeviceModel::getTileType` assertion으로 abort. 스택
   트레이스가 심볼 없이 raw 주소만 찍힘 (`LLVM_SYMBOLIZER_PATH`를 잡아줘도 이 빌드에서는
   symbolizer 바이너리 자체가 없었음).
2. attention 블록만 떼서 export → 동일 크래시. 두 개 encoder layer 전체 → 동일 크래시.
3. 손으로 최소 ONNX(`two_bmm.onnx`: batched matmul 2개만)를 만들어 재현 → 동일 크래시.
   처음엔 "batched matmul을 두 개 연달아 컴파일하면 깨진다"로 가설을 세움.
4. 그런데 두 번째 matmul(Attn@V 모양)만 **단독으로** 컴파일해도 크래시하는 걸 확인 →
   가설 기각. "두 개를 체인" 문제가 아니라 **특정 shape** 문제였다.
5. `M,K,N`을 하나씩 바꿔가며 이분 탐색: `M=16,K=64,N=16`(QK^T 모양)은 항상 성공,
   `M=16,N=64`(K값 무관, 16이든 32든)는 항상 크래시, **`M=32,N=64`는 성공**. `seq_len`이
   곧 이 M/N이므로 — seq_len을 32로 올리면 되겠다는 결론.
6. 정확한 호출 지점은 스택 주소를 직접 심볼로 복원해서 확인: 크래시난 바로 그
   프로세스의 `/proc/<pid>/maps`에서 `libIREECompiler.so` 로드 베이스를 읽고, 크래시
   메시지의 절대주소에서 베이스를 빼서 파일 상대주소를 구한 뒤, `nm -C
   build/lib/libIREECompiler.so`로 뽑은 정렬된 심볼 테이블에서 가장 가까운 심볼을
   찾는 방식(수동 addr2line)으로 `AMDAIEGenerateControlOverlayPass::runOnOperation` →
   `generateControlOverlay` → `getTileType`까지 특정했다.
7. `--mlir-print-ir-after-all --mlir-print-ir-after-change --mlir-disable-threading`로
   pass별 IR 덤프를 받아서, 크래시 직전 마지막으로 성공한 pass(`AMDAIEAccessToAcquireRelease`)
   와 그 안의 `amdaie.tile` 배정 값(row 0~9)을 직접 눈으로 확인 — §5의 스니펫이 바로 이것.

## 7. 최종 결과

`SEQ_LEN`을 16 → 32로 바꾼 뒤:

```
compile: iree-compile ... (VGG16 레시피에서 im2col/channels-last/detach-elementwise 3개
          conv 전용 플래그만 제거) → exit 0
run:     iree-run-module --device=amdxdna --device=local-task ... → exit 0  (실제 npu4에서 실행)
verify:  corr(torch reference, npu4 output) = 0.99998, max abs diff = 0.029
```

Q/K/V/output/FFN projection과 QK^T, Attn@V는 전부 NPU, embedding/LayerNorm/Softmax/GELU는
전부 CPU. 목표했던 배치 그대로 실제 하드웨어에서 재현됐다.

## 8. 남은 문제

- `seq_len < 32` (적어도 attention의 `M=N=seq_len`이 16처럼 작고 그 N에 대한 N-타일
  개수가 물리 row 수를 넘는 조합)에서는 여전히 크래시한다. 근본 수정은
  `AMDAIEAssignTiles`/`generateControlOverlay` 쪽에서 N-타일이 num_rows를 넘어갈 때
  다음 컬럼으로 넘기도록(또는 최소한 크래시 대신 진단 메시지를 내도록) 고쳐야 하는데,
  이번 세션에서는 하지 않았다.
- 이 문서의 root-cause 확인은 디버그 심볼이 없는 상태에서 수작업으로 스택을 복원한
  결과다. 정식으로 고치려면 `-DCMAKE_BUILD_TYPE=RelWithDebInfo` 등으로 다시 빌드해서
  gdb/lldb로 정식 백트레이스를 뜨는 게 먼저다.

관련 자세한 사용법·플래그 표는 `models/bert_tiny/README.md`, 세션 간 기록은
Claude 메모리(`project-bert-e2e-plan`)에 남아 있다.
