# 네이티브 conv 의 시간 분할과 공간 분할 — 차원마다 왜 이렇게 나누는가

> 작성 2026-11-09 · 브랜치 `conv-npu`(HEAD `786d1d19` + 미커밋 native conv 작업 트리) · 컴파일러 `build/lib/libIREECompiler.so` 11-09 10:53 빌드(소스 최신 수정보다 나중) · 타겟 npu4
>
> 요청: "native conv 파이프라인(im2col 을 쓰지 않는 conv2d 하강 경로)에서 각 단계의 시간 분할과 공간 분할 알고리즘을 파악해, 논리적이고 직관적으로 설명해 줘. 핵심 질문은 '각 차원(N, OH, OW, OC, IC, KH, KW)을 왜 이렇게 나누고, 왜 이렇게 분배하는가' 이다."
>
> 검증 범위: 컴파일과 정적 IR(패스 덤프)만 했다. NPU 에서는 실행하지 않았다. 따라서 수치 정확도와 실행 시간은 모른다.
>
> 관련 문서: 알고리즘 전체 서술은 [1109_native_conv_algorithm.md](1109_native_conv_algorithm.md), 구현 경과는 [1108_native_conv_impl.md](1108_native_conv_impl.md).

## 0. 결론 먼저

### 0.1 이 브랜치에는 im2col 이 아닌 conv 경로가 둘 있다

| | native conv 타일 파이프라인 (`conv`) | conv-decompose (예전 경로) |
|---|---|---|
| 켜는 법 | `--iree-amdaie-enable-native-conv` (`run-onnx.sh --conv native`) | `--iree-amdaie-tile-pipeline=conv-decompose` |
| 받는 conv | 6 루프(배치 1 이 접힌 것) 또는 7 루프 generic (`getConvDims`, `AMDAIEUtils.cpp:966`) | named conv 또는 반복자 7 개 generic (`AMDAIEUtils.cpp:590-599`) |
| 배열 | 최대 8 열 × 4 행 = 32 코어 | 1 열 × 4 코어 |
| IC | L1 로 조금씩 흘려 넣음 | L1 에 전부 올림 |
| npu4 IR 결과 | 대표 모양 6 개(VGG 층 5 + OH 7) 모두 컴파일, 감사 PASS | 벡터화를 끈 작은 N=2 conv 만 됨. IC 512 는 BD 8,244 개로 실패, L1 도 약 209 KB 필요 |

- 실제 모델(ONNX, 배치 1)의 conv 는 IREE 가 배치를 접어 6 루프로 넘긴다. conv-decompose 는 이것을 받지 못한다.
- 그래서 지금 "native conv" 는 `conv` 파이프라인을 말한다.
- 메모리에 남은 사실들("열 1 개", "IC 512 에서 L1 이 넘친다", "BD 7,224 개")은 모두 conv-decompose 경로의 이야기다. 비교는 §6 에 있다.
- 이 문서의 나머지는 `conv` 파이프라인을 설명한다.

### 0.2 차원마다 세 몫으로 나뉜다

`conv` 파이프라인에서 차원 하나는 최대 세 몫으로 쪼개진다.

- **공간 몫**: 배열의 열·행에 동시에 놓인다.
- **시간 몫**: 블록 루프나 채널 루프로 차례로 돈다.
- **코어 안 몫**: 코어가 L1 위에서 도는 스칼라 루프다.

| 차원 | 공간 | 시간 | 코어 안 | 한 줄 이유 |
|---|---|---|---|---|
| N (배치) | — | — | — | 배치 1 은 IREE 가 접어 없앤다. 그 자리를 OH 를 자른 청크 n 이 쓴다 (§3.5) |
| OH | **열** (청크 n 개, 청크마다 S 행) | 블록 루프 (S / ohTile 번) | ohTile 행 | 열마다 memtile 하나가 자기 청크의 입력만 받고 출력만 모은다 (§3.3, §3.4) |
| OW | — | 블록 루프 (OW / owTile 번) | owTile 열 | 공간 축 둘은 청크와 OC 가 이미 쓴다 (§3.1) |
| OC | **행** (4 행 × 16 채널) | 블록 루프 (OC / 64 번) | 16 채널 | 입력은 열 방향, 필터는 행 방향으로 방송해 둘 다 공유한다 (§3.2) |
| IC | — | 채널 루프 (IC′ / icTile 번) | icTile 채널 | 누산 축이다. 누산기를 L1 에 두고 채널을 흘려 넣는다 (§4.2) |
| KH, KW | — | — | 3 × 3 전부 | 출력을 자르면 창(halo)이 따라온다. 부분합이 생기지 않는다 (§4.3) |

IC′ 는 bias 를 접은 뒤의 입력 채널 수다. IC + 1 을 8 의 배수로 올린다(§2.1).

### 0.3 대표 층의 분해

IR 로 확인했다(§5). 각 칸의 곱이 그 차원의 크기다.

| 층 (출력² · IC → OC) | OH = 열 × 블록 × 코어 | OW = 블록 × 코어 | OC = 행 × 블록 × 코어 | IC′ = 단계 × icTile | 코어 |
|---|---|---|---|---|---|
| conv1_2 (224² · 64 → 64) | 8 × 2 × 14 | 16 × 14 | 4 × 1 × 16 | 9 × 8 | 32 |
| conv3_2 (56² · 256 → 256) | 8 × 1 × 7 | 2 × 28 | 4 × 4 × 16 | 33 × 8 | 32 |
| conv4_1 (28² · 256 → 512) | 7 × 1 × 4 | 1 × 28 | 4 × 8 × 16 | 11 × 24 | 28 |
| conv4_2 (28² · 512 → 512) | 7 × 1 × 4 | 2 × 14 | 4 × 8 × 16 | 13 × 40 | 28 |
| conv5_x (14² · 512 → 512) | 7 × 1 × 2 | 1 × 14 | 4 × 8 × 16 | 13 × 40 | 28 |
| 7² · 64 → 64 | 4 × 1 × 2 (8 행 중 1 행은 패딩) | 1 × 7 | 4 × 1 × 16 | 1 × 72 | 16 |

- 얕은 층에서는 화면 위치(OH·OW)가 시간 축이다.
- 깊은 층에서는 OC 가 시간 축이다.
- 공간 타일이 작아질수록 L1 에 남는 자리만큼 icTile 이 커진다(8 → 24 → 40).

## 1. 결정의 근거가 되는 제약

| 제약 | npu4 값 | 출처 | 영향 |
|---|---|---|---|
| 배열 모양 | 8 열 × 4 코어 행. 열마다 memtile 1·shim 1 | `iree_aie_runtime.cc:864`, 실행 대상 설정 `num_cols = 8, num_rows = 4` | 공간 축은 열과 행 둘뿐 |
| 코어 L1 | 64 KB. 스택 1 KB 를 빼고 64,512 B | aie-rt `xaie2pgbl_reginit.c:185`, `AIETarget.h:145`, `KernelDispatch.cpp:820` | 코어 타일과 icTile 의 상한 |
| memtile L2 | 512 KB | `xaie2pgbl_reginit.c:2201` | 여유가 크다. 실제 사용은 136 KB 이하 |
| 코어 DMA 채널 | 받기(S2MM) 2, 내보내기(MM2S) 2 | `xaie2pgbl_reginit.c:1911` | 받는 스트림이 입력·필터 둘뿐이다. 그래서 bias 를 conv 에 접고, 누산기는 L1 에 둔다 |
| memtile DMA 채널 | S2MM 6, MM2S 6 | `xaie2pgbl_reginit.c:1673` | memtile 하나가 모을 수 있는 생산자 수를 정한다 |
| shim DMA 채널 | S2MM 2, MM2S 2 | `xaie2pgbl_reginit.c:2164` | 열마다 입력 1 + 필터 1 |
| BD 개수 | 코어 16, memtile 48, shim 16 | `xaie2pgbl_reginit.c:1897, 1659, 2150` | 단계·블록 반복을 BD 의 반복 차원으로 접어야 한다 |
| shim BD 반복 차원 | 63 이하 | `xaie2pgbl_reginit.c:2066` (IterWrapMax) → `AMDAIEUtils.cpp:1080` | 블록 하나의 IC 단계를 BD 하나에 접으려면 단계 수 ≤ 63 |
| DMA 주소·길이 | 4 B 단위 | `xaie2pgbl_reginit.c:2059` (AddrAlignMask 0x3), `AMDAIEUtils.cpp:1082-1088` | 자르는 모든 차원에서 4 B 경계를 지킨다 |
| 코어 버퍼 시작 | 64 B 정렬 (64 B 미만 버퍼는 32 B) | `AMDAIEUtils.cpp:1040-1046` | L1 계산에 정렬을 넣는다 |
| 버퍼 깊이 | L2·L1 모두 2 (이중 버퍼) | `Passes.td:178-180` | L1 = 2 × (입력 + 필터 + 누산기 + epilogue) |
| 코어 forall | 반복 수가 1 보다 큰 차원은 2 개까지 | `AMDAIETileAndFuse.cpp:62-86` | 공간으로 퍼지는 차원은 정확히 둘 |
| 블록 forall | 시간으로 펼쳐진다 | `AMDAIETileAndFuse.cpp:385-386` ("blocks get unrolled temporally") | 블록 = 시간 축 |
| 코어 좌표 | 열 = thread x, 행 = thread y + 2 | `AMDAIEInsertCores.cpp:141-143` | forall 의 첫 차원을 열에 두려면 x/y 를 바꾼다 (`swapThreadXY`) |
| memtile 분할 | L2 버퍼를 **첫 비단위 차원**으로 자른다. 조각 수 = gcd(소비자 또는 생산자 수, 쓰는 열 수) | `AMDAIESplitLogicalObjFifos.cpp:201-206, 272, 342, 396-400` | 열마다 나눌 차원을 맨 앞에 둬야 한다 |
| 단위 차원 접기 | IREE 가 크기 1 인 차원을 접는다 | `AMDAIEChunkConvDispatches.cpp:41-44` | 청크당 행 S ≥ 2 |

## 2. 단계별 흐름과 메모리 단계

```
L3 (DDR, host 쪽 Flow 재작성)
  ① bias 접기: IC → IC′ = alignTo(IC + 1, 8)
  ② 행 청크: 입력 [N][S+2][W+2][IC′] 를 host gather 로 만들고, 필터는 FHWC [OC][3][3][IC′]
NPU dispatch
  ③ 전략: chooseConvTiling → tile_sizes = [[블록], [채널], [코어]]
  block forall                (시간)  청크 nCores 개 × ohTile × owTile × (4 × 16) OC     L3 ↔ L2
    channel scf.for           (시간)  icTile 채널씩                                       L3 → L2 → L1
      core forall             (공간)  청크 → 열, OC 타일 → 행                             L2 → L1
        core                  (L1)    ohTile × owTile × 16 출력에 3 × 3 × icTile 누산      스칼라 루프
L3 (host)
  ④ [N][S][OW][OC] 를 [N·S][OW][OC] 로 reshape → 패딩 행 자르기 → (다음 층 버퍼에 insert)
```

### 2.1 Flow 단계 (host, L3)

**bias 접기** (`AMDAIEFoldBiasIntoContractionDispatches.cpp:32-41, 595`)
- 코어가 받을 수 있는 스트림은 입력과 필터 둘뿐이다. bias 를 세 번째 입력으로 둘 수 없다.
- 그래서 입력에 1 로 채운 채널을 붙이고, 필터의 그 채널 가운데 탭에 bias 를 넣는다.
- 채널 수는 IC + 1 을 8 의 배수로 올린다(64 → 72, 256 → 264, 512 → 520).
- 8 의 배수인 까닭: icTile 은 8 의 배수여야 한다(`AMDAIEUtils.cpp:1142`). NHWC 에서 픽셀 하나의 채널 조각이 16 B 이상 연속으로 움직이게 하려는 규칙이다(`AMDAIEUtils.h:248-249`). IC′ 가 8 의 배수가 아니면 8 의 배수 icTile 로 IC′ 를 나눌 수 없다.

**행 청크** (`AMDAIEChunkConvDispatches.cpp:189-193`)
- S = max(2, ⌈OH / 8⌉), N = ⌈OH / S⌉ 이다. N < 2 이면 나누지 않는다(`:520-527`).
- host gather 가 청크 n 에 입력 행 [n·S, n·S + S + 2) 를 모은다(`:262-344`). 이웃 청크와 2 행(KH − 1)이 겹친다.
- 상수 HWCF 필터는 FHWC 로 바꾼다(`:197-218`).
- 출력 [N][S][OW][OC] 는 `flow.tensor.reshape` 로 [N·S][OW][OC] 가 된다(`:596-600`). 패딩이 있으면 `flow.tensor.slice` 로 앞 OH 행만 남긴다(`:601-613`).

### 2.2 전략 (`KernelDispatch.cpp:744-856`, `AMDAIEUtils.cpp:1048-1164`)

tile_sizes 는 루프 (n, oh, ow, oc, kh, kw, ic) 에 대해 세 단계다(`KernelDispatch.cpp:831-845`).

| 단계 | 값 | 뜻 |
|---|---|---|
| 0 블록 | [nCores, ohTile, owTile, 4 · ocTile, 0, 0, 0] | 블록 하나 = 모든 코어의 타일 한 벌 |
| 1 채널 | [0, 0, 0, 0, 0, 0, icTile] | 채널 루프 한 단계 |
| 2 코어 | [1, 0, 0, ocTile, 0, 0, 0] | 코어 하나 = 청크 1 개 × OC 타일 1 개 |

- 블록의 공간 크기(ohTile × owTile)는 코어 타일과 같다. 블록 = 코어 타일의 합집합이다. L2 에서 코어 타일보다 큰 단위로 재사용하는 단계는 없다.
- 탐색 순서(`AMDAIEUtils.cpp:1117-1162`):
  1. ocCores(쓰는 행 수)를 4 부터 줄여 가며, 가능한 타일이 하나라도 있으면 거기서 멈춘다. 코어를 가장 많이 쓰는 배치가 먼저다.
  2. 그 안에서 ocTile | OC/ocCores, ohTile | S, owTile | OW, icTile | IC′ 를 모두 본다.
  3. 4 B 정렬(`cutsAligned`), icTile % 8 == 0, L1 ≤ 64,512 B 를 지키는 것만 남긴다.
  4. 순위로 고른다: (IC′/icTile ≤ 63, 코어가 받는 바이트당 MAC, icTile, owTile). 수치 예는 §5.

### 2.3 타일 파이프라인 (`Passes.cpp:451-588`)

- 블록 forall(level 0, `:466-474`)을 만들고 첫 복사 단계를 넣는다. 블록의 출력 버퍼를 L2 에 잡는다(`:476-500`).
- 채널 루프가 생기기 **전에** 두 번째 복사 단계를 넣는다(`:502-508`). 그래서 블록의 누산기가 채널 루프 내내 L1 에 머문다.
- 채널 루프(level 1, `scf.for`, `:512-520`)를 만들고, 입력·필터 복사를 그 안으로 넣는다(`:522-531`). 입력·필터의 L2 버퍼는 이때 IC 조각 하나 크기로 잡힌다(`:533-542`).
- 코어 forall(level 2, `:544-557`)은 `swapThreadXY` 로 첫 차원(청크)을 열에, 둘째(OC)를 행에 둔다.
- 채널 루프를 벗긴다(`:579-581`). 첫 단계에는 0 채우기를, 마지막 단계에는 ReLU 를 붙인다.
- 코어 코드는 스칼라 루프다. 마이크로커널이 없고, native 레시피는 벡터화를 끈다.

### 2.4 objectFifo 하강

- `AMDAIEConvOutputSlicesInOrder`(`Passes.cpp:1116-1120`)가 L2 출력 배치를 [slice][oh][ow][16] 으로 바꾼다. 코어마다 연속 구간을 쓰게 되어, 생산자마다 lock 쌍을 따로 갖는다(`AMDAIEConvOutputSlicesInOrder.cpp:7-25`).
- `AMDAIESplitLogicalObjFifos`(`Passes.cpp:1152-1158`)가 L2 버퍼를 열마다의 memtile 로 나눈다. 이 패스는 전역 파이프라인이 pack-peel·general-copy 일 때만 돈다. native 레시피의 전역 파이프라인은 pack-peel 이라 돈다.
- control code 에서 블록 forall 은 `scf.for` 가 된다. 블록 하나의 IC 단계들은 shim BD 하나의 반복 차원으로 접힌다(§4).

## 3. 공간 분할 — 왜 (청크 → 열, OC 타일 → 행) 인가

### 3.1 공간 축은 둘뿐이고, 누산 축은 퍼뜨리지 않는다

- 배열이 2 차원이라 코어 forall 이 퍼뜨릴 수 있는 차원은 둘이다(`AMDAIETileAndFuse.cpp:62-86`). 일곱 차원 중 둘만 공간으로 가고, 나머지는 시간이나 코어 안으로 간다.
- 누산 차원(IC, KH, KW)은 공간으로 퍼뜨리지 않는다.
  - 퍼뜨리면 같은 출력의 부분합이 여러 코어에 생기고, 이를 더할 경로가 필요하다.
  - 코어가 받을 수 있는 스트림은 입력·필터 둘이 전부다. 부분합을 받을 세 번째 채널이 없다.
  - 그래서 출력 원소 하나는 코어 하나가 맡고, 9 · IC′ 개의 곱을 자기 L1 에서 모두 더한다(출력 고정 방식).
- 남은 후보는 병렬 차원인 OH(청크), OW, OC 셋이다. 그중 청크와 OC 가 고정된다.

### 3.2 OC 를 행에 둔다 — 두 피연산자를 모두 방송으로 나눠 쓴다

- 한 열의 코어 4 개는 같은 청크의 같은 화소를 계산하고, 출력 채널만 다르다. 그러니 같은 입력 창이 필요하다.
  - memtile c 가 입력 창을 한 번 보내고, 그 열의 코어 4 개가 함께 받는다.
  - IR: 입력 L1 objectFifo 가 `{tile_c_2, tile_c_3, tile_c_4, tile_c_5}` 에 놓인다.
- 한 행의 코어들은 같은 OC 타일을 서로 다른 청크에 대해 계산한다. 그러니 같은 필터 조각이 필요하다.
  - 필터 조각 하나를 그 행의 코어 7–8 개가 함께 받는다.
  - IR: 필터 L1 objectFifo 가 `{tile_0_r, …, tile_6_r}` 에 놓인다. 행 2 가 OC 블록의 0–15 번 채널, 행 5 가 48–63 번 채널을 받는다.
- 이렇게 하면 L2 → L1 스트림 하나가 코어 4 개나 7–8 개를 먹인다. 입력은 열 방향, 필터는 행 방향으로 흐르는 2 차원 배치다.
- OW 를 행에 두면 한 열의 코어 4 개가 서로 다른 입력 창을 받아야 해 입력 방송이 사라진다. (설계 논리이며, 이 배치를 컴파일해 보지는 않았다.)
- 필터는 FHWC 여야 한다.
  - FHWC 에서는 OC 타일 하나가 필터의 연속 구간이다.
  - HWCF 에서는 행들의 OC 타일이 번갈아 놓인다. memtile 분할은 첫 차원(KH)으로 자르고, 공유 카운팅 lock 은 경쟁 상태가 된다.
  - 그래서 HWCF 필터는 행을 1 개만 쓴다(`AMDAIEUtils.cpp:1068-1078`). 청크 패스가 상수 HWCF 필터를 FHWC 로 바꾸는 까닭이다.
  - IR: 배치 2 모델은 청크 패스를 거치지 않아 HWCF(`3x3x72x64`)가 그대로 남는다. 결과는 열 2 개 × 행 1 개 = 코어 2 개다(§3.5).

### 3.3 청크를 열에 둔다 — memtile 받기 채널 6 개의 셈

memtile 하나(열마다 하나)가 받아야 하는 것은 입력 1 스트림, 필터 1 스트림, 그리고 그 memtile 에 출력을 쓰는 코어들의 스트림이다.

| 배치 | memtile 하나에 출력을 쓰는 코어 | 받기 채널 | 가능? |
|---|---|---|---|
| G1: 청크 → 열, OC → 행 (지금) | 그 열의 코어 4 개 | 1 + 1 + 4 = **6** | 한계와 같음 |
| G2: OC → 열, 청크 → 행 | 같은 청크를 맡은 한 행의 코어 8 개 | 1 + 1 + 8 = 10 | 6 을 넘음 |

- G2 가 안 되는 까닭은 memtile 분할이 L2 출력 [n][oh][ow][oc] 를 첫 차원 n(청크)으로 자르기 때문이다. memtile 하나가 청크 하나의 모든 OC 를 갖게 되고, 그 OC 를 나눠 맡은 8 열의 코어가 모두 이 memtile 에 써야 한다(`AMDAIEUtils.h:214-229`).
- IR: 컴파일한 모델 6 개 모두 memtile S2MM 사용이 6/6 이다.
- 보내기(MM2S) 쪽은 열 수에 따라 갈린다.
  - 8 열(·4 열) 층: memtile 0–3 이 필터를 OC 16 개씩 나눠 든다(gcd(4 행, 8 열) = 4). 입력 → 자기 열, 필터 → 한 행, 출력 → shim 으로 3 채널을 쓴다.
  - 7 열 층: gcd(4, 7) = 1 이라 필터가 나뉘지 않는다. memtile (0,1) 이 OC 블록 64 개를 모두 들고 네 행에 따로 보낸다. 입력 1 + 필터 4 + 출력 1 = **6/6** 이다.

### 3.4 OH 를 청크로 자른다 — 왜 OW 가 아니라 OH 인가

- NHWC 에서 H 가 가장 바깥 차원이다. 행 묶음은 DDR 에서 연속 구간이고, halo 는 통째 행 KH − 1 개다. gather 는 `n·S + r` 행 지도 하나로 된다(`AMDAIEChunkConvDispatches.cpp:305-309`).
- 출력을 되돌리는 일이 공짜다. 행 청크들을 이어 붙이면 원래 행 순서이므로 [N][S][OW][OC] → [N·S][OW][OC] 는 복사 없는 reshape 다(`:596-600`). OW 로 자르면 되돌릴 때 전치 복사가 필요하다.
- memtile 분할은 첫 차원으로 자른다. 그래서 청크 차원을 맨 앞에 둔 [N][S+2][W+2][IC′] 를 명시적으로 만든다(`:18-24`).
- 청크 크기 S 를 정하는 논리(`:41-49`):
  - 모든 열이 자기 청크를 동시에 계산하므로 층 시간은 S 에 비례한다. 그래서 S 를 먼저 최소로 한다: S = ⌈OH / 8⌉.
  - 단, S ≥ 2 다. 크기 1 인 OH 차원은 IREE 가 접어 conv 로 인식되지 않는다.
  - 그 S 로 OH 를 덮는 가장 적은 청크 수 N = ⌈OH / S⌉ 를 쓴다. N · S > OH 이면 남는 행은 0 을 읽고 계산한 뒤 잘라낸다.
- 예:
  - OH 224 / 56 → 8 × 28 / 8 × 7. 패딩 없음, 32 코어.
  - OH 28 → 7 × 4. 8 열이어도 S 는 ⌈28/8⌉ = 4 이므로 8 번째 열은 패딩 행만 맡게 된다. 그래서 7 열을 쓴다.
  - OH 14 → 7 × 2.
  - OH 7 → 4 × 2. 1 행 패딩. IR 에 `flow.tensor.slice … tensor<8x7x64xf32> -> tensor<7x7x64xf32>` 가 있다.
- 비용은 halo 중복이다. 입력 행 수가 (S + 2) / S 배가 된다: S = 28 이면 1.07 배, S = 2 이면 2 배.

### 3.5 N (배치)

- 배치 1: IREE 가 배치를 접어 6 루프 conv 로 넘긴다. 그 자리를 청크 차원 n 이 쓴다.
- 배치 > 1: n 이 진짜 배치가 된다.
  - 청크 패스는 7 루프 conv 를 건너뛴다(`AMDAIEChunkConvDispatches.cpp:125`).
  - 열 수 nCores 는 N 의 약수 중 8 이하인 가장 큰 값이다(`AMDAIEUtils.cpp:1063-1067`).
  - 필터가 HWCF 로 남아 행은 1 개다.
  - IR(`pc_b2_28x64x64`): tile_sizes `[[2, 14, 14, 16, 0, 0, 0], [0, 0, 0, 0, 0, 0, 8], [1, 0, 0, 16, 0, 0, 0]]`, 코어 2 개(2 열 × 1 행), OC 64 는 시간으로 4 블록이다.

## 4. 시간 분할 — 블록 루프, 채널 루프, 코어 안

### 4.1 블록 루프 (L3 ↔ L2)

- 블록 forall 은 시간으로 펼쳐진다(`AMDAIETileAndFuse.cpp:385-386`). control code 에서는 (oh 블록, ow 블록, oc 블록) 순서의 `scf.for` 다. n 은 모두 공간이라 루프가 없다.
- IR(c42 = conv4_2, control code):

  ```
  for ow 블록 (2) {
    for OC 블록 (8) {
      열 c 마다 입력 BD 1 개: [13 단계][6 행][16 열][40 채널], stride [40, 15600, 520, 1]
      shim (0,0) 필터 BD 1 개: [13 단계][64 OC][9 탭][40 채널]
      모두 기다림
      열 0..6 의 출력 BD 를 하나씩 내고 기다림: [4 조각][4][14][16] f32
    }
  }
  ```

- IR(c12 = conv1_2): `for oh 블록 (2) { shim 0–3 필터 BD: [16 반복(stride 0)][9 단계][144][8]; for ow 블록 (16) { 8 열 입력 BD [9][16][16][8] → 대기 → 출력 BD 8 개 } }`. 필터는 ow 와 무관하므로 16 블록을 stride 0 반복 하나로 접었다.
- L2 에는 지금 블록의 IC 조각 하나(이중 버퍼)만 있다. 그래서 블록 사이에 재사용이 없다.
  - 입력 창은 OC 블록마다 DDR 에서 다시 읽는다.
  - 필터는 공간 블록마다 다시 읽는다.

| 층 | 블록 = 공간 × OC | IC 단계 | 입력 DDR 읽기 (gather 결과 대비) | 필터 DDR 읽기 (필터 크기 대비) | 출력 쓰기 (f32) |
|---|---|---|---|---|---|
| conv1_2 | 32 = 32 × 1 | 9 | 9.44 MB (1.21 배) | 2.65 MB (32 배) | 12.85 MB |
| conv3_2 | 8 = 2 × 4 | 33 | 9.12 MB (4.14 배) | 2.43 MB (2 배) | 3.21 MB |
| conv4_1 | 8 = 1 × 8 | 11 | 5.32 MB (8.00 배) | 2.43 MB (1 배) | 1.61 MB |
| conv4_2 | 16 = 2 × 8 | 13 | 11.18 MB (8.53 배) | 9.58 MB (2 배) | 1.61 MB |
| conv5_x | 8 = 1 × 8 | 13 | 3.73 MB (8.00 배) | 4.79 MB (1 배) | 0.40 MB |

- 식: 입력 = 블록 수 × N × (ohTile+2)(owTile+2) · IC′ · 2 B, 필터 = 공간 블록 수 × OC · 9 · IC′ · 2 B.
- conv4_2 와 conv1_2 는 control code 의 BD 크기를 곱한 값과 같다(c42: 16 블록 × 7 열 × 99,840 B = 11.18 MB).
- 얕은 층은 필터를 여러 번 읽지만 필터가 작다. 깊은 층은 입력을 OC 블록 수만큼 다시 읽는다.

### 4.2 채널 루프 (IC, L3 → L2 → L1)

- 채널 루프는 코어 forall **밖**의 `scf.for` 다(`Passes.cpp:512-520`). 블록의 누산기는 루프 내내 L1 에 머문다(`:502-508`).
- IC 전체를 L1 에 올리지 않는 까닭: L1 은 64 KB 다. conv4_2 에서 코어 하나의 필터 조각만 해도 16 OC × 9 × 520 × 2 B = 149,760 B 다.
- 단계마다 부분합을 L2 로 내보냈다 다시 읽지 않는 까닭: 코어에 세 번째 받기 채널이 필요하다(코어 S2MM 2 개). GEMV 파이프라인에 같은 설명이 있다(`Passes.cpp:363-366`).
- control code 에서 블록 하나의 IC 단계들은 BD 의 가장 바깥(반복) 차원이 된다. c42 의 입력 BD [13][6][16][40] 이 그 예다. shim BD 의 반복 차원은 63 이하라, 순위의 첫 기준이 IC′ / icTile ≤ 63 이다.
- 채널 루프를 벗긴다(`Passes.cpp:579-581`). 첫 단계는 누산기를 0 으로 채우고, 마지막 단계는 ReLU 를 출력 타일에 쓴다.

### 4.3 코어 안 (L1)

- 단계 하나는 [ohTile][owTile][16] 출력 × 9 탭 × icTile 채널의 곱-누산이다. 스칼라 루프다.
- KH·KW 는 코어 op 안에 남는다(`Passes.cpp:452-459`). 출력을 자르면 각 단계가 필요한 입력 창((ohTile+2) × (owTile+2))을 스스로 받는다.
- KH·KW 를 자르면 같은 출력에 탭별 부분합이 생긴다. 3 × 3 은 작아서 자를 이득도 없다.
- 코어 하나가 도는 양:

| 층 | 코어당 커널 호출 (블록 × 단계) | 호출당 MAC | 코어당 MAC |
|---|---|---|---|
| conv1_2 | 32 × 9 = 288 | 225,792 | 65.0 M |
| conv3_2 | 8 × 33 = 264 | 225,792 | 59.6 M |
| conv4_1 | 8 × 11 = 88 | 387,072 | 34.1 M |
| conv4_2 | 16 × 13 = 208 | 322,560 | 67.1 M |
| conv5_x | 8 × 13 = 104 | 161,280 | 16.8 M |

c12·c42 의 호출 수(288, 208)는 `conv_dma_check.py` 가 IR 에서 센 L1 fill 수와 같다.

## 5. 크기 결정 — chooseConvTiling 을 숫자로 따라가기

### 5.1 L1 식

코어 하나의 L1 (a = 64 B 정렬, bf16 입력·필터, f32 누산기·출력):

L1 = 2 × [ a((ohTile+2)(owTile+2) · icTile · 2) + a(9 · 16 · icTile · 2) + a(ohTile · owTile · 16 · 4) × 2 ] ≤ 64,512 B

- 마지막 항은 f32 누산기와 f32 epilogue 출력이다. icTile 과 무관하고 크다.
- 14 × 14 × 16 타일이면 이 항만 2 × 2 × 12,544 = 50,176 B 로 L1 의 78 % 다. 그래서 얕은 층의 icTile 은 8 이다.
- 실제 배치는 누산기 1 · 임시 1 · 출력 2 개다(conv1_2 코어 (0,2): f32 12,544 B × 4, 입력 4,096 B × 2, 필터 2,304 B × 2 = 62,976 B, 스택 포함 끝 주소 64,000 B).

### 5.2 순위

1. IC′ / icTile ≤ 63. 블록의 IC 단계를 shim BD 하나에 접을 수 있어야 한다.
2. 코어가 받는 바이트당 MAC = 9 · ohTile · owTile · 16 / (9 · 16 · 2 + (ohTile+2)(owTile+2) · 2). 채널 하나당 값이라 icTile 과 무관하다. 코어의 받기 채널 2 개의 대역폭을 아끼는 기준이다.
3. 큰 icTile. 단계 수와 lock 왕복이 줄어든다.
4. 큰 owTile.

### 5.3 예: 왜 conv1_2 는 14 × 14, conv4_2 는 4 × 14 인가

**conv1_2** (S = 28, OW = 224, IC′ = 72, OC = 64)
- OC = 64 를 4 행에 나누면 ocTile 은 16 이 최대다.
- 14 × 14 × 16 의 MAC/B 는 35.3 이고, 7 × 28 × 16 은 34.1 이다. 정사각형에 가까울수록 halo 비율이 작다.
- 14 × 28 이나 28 × 14 는 누산기만 25,088 B 라, × 2 (누산·출력) × 2 (이중 버퍼) = 100 KB 로 넘친다.
- 남은 L1 에 들어가는 icTile 은 8 이다 → 9 단계.

**conv4_2** (S = 4, OW = 28, IC′ = 520 = 2³ · 5 · 13, OC = 512)
- 8 의 배수인 IC′ 의 약수는 8, 40, 104, 520 뿐이다.
- 4 × 28 × 16 이면 L1 식에서 icTile ≤ 27 → 8 → 단계 65 개 > 63. 순위 1 번에서 진다.
- 4 × 14 × 16 이면 icTile ≤ 52 → 40 → 13 단계. 이것이 뽑힌다.
- 즉 shim BD 반복 한계 63 이 공간 타일을 좁혔다.
- conv4_1 은 IC′ = 264 의 약수에 24 가 있다. 4 × 28 × 16 에서 icTile 24 → 11 단계라 4 × 28 을 유지한다.

**ocTile 이 늘 16 인 까닭**
- 행을 4 개 다 쓰는 배치가 먼저라 ocTile ≤ OC / 4 다. OC = 64 이면 정확히 16 이다.
- OC = 512 이면 더 큰 ocTile 도 가능하다. 그러나 필터 조각(9 · ocTile · icTile · 2 B)이 커져 공간 타일이 작아지고 MAC/B 가 떨어진다. 예: conv5 의 2 × 7 × 32 는 6.2, 2 × 14 × 16 은 9.7.

### 5.4 IR 확인 표

단일 층 ONNX 모델 6 개(+ 배치 2)를 `run-onnx.sh compile --conv native --numeric bf16 --vectorize off --ukernel all --pool cpu` 로 컴파일했다. 모두 rc = 0, `aie_audit.py` PASS 다. 결과 디렉터리는 `build/onnx/pc_*-conv_native-ukernel_all-vectorize_off-numeric_bf16-pool_cpu/` 이고, `config.json` 의 commit 은 `786d1d19`(+ 미커밋 트리)다.

| 모델 | Flow 디스패치: 입력 / 필터 / 출력 | tile_sizes (블록 / 채널 / 코어) | 열 · 코어 | 코어 (0,2) L1: 합 / 끝 주소 | memtile (0,1) | memtile S2MM / MM2S |
|---|---|---|---|---|---|---|
| c12 224²·64→64 | 8×30×226×72 / 64×3×3×72 / 8×28×224×64 | [8,14,14,64] / ic 8 / [1,·,·,16] | 8 · 32 | 62,976 / 64,000 | 113,152 B | 6 / 3 |
| c32 56²·256→256 | 8×9×58×264 / 256×3×3×264 / 8×7×56×256 | [8,7,28,64] / 8 / [1,·,·,16] | 8 · 32 | 63,424 / 64,512 | 113,600 B | 6 / 3 |
| c41 28²·256→512 | 7×6×30×264 / 512×3×3×264 / 7×4×28×512 | [7,4,28,64] / 24 / [1,·,·,16] | 7 · 28 | 59,776 / 60,800 | 129,920 B | 6 / 6 |
| c42 28²·512→512 | 7×6×30×520 / 512×3×3×520 / 7×4×28×512 | [7,4,14,64] / 40 / [1,·,·,16] | 7 · 28 | 52,736 / 60,416 | 136,192 B | 6 / 6 |
| c52 14²·512→512 | 7×4×16×520 / 512×3×3×520 / 7×2×14×512 | [7,2,14,64] / 40 / [1,·,·,16] | 7 · 28 | 40,448 / 56,064 | 116,736 B | 6 / 6 |
| c7 7²·64→64 | 4×4×9×72 / 64×3×3×72 / 4×2×7×64 | [4,2,7,64] / 72 / [1,·,·,16] | 4 · 16 | 55,424 / 60,480 | 59,008 B | 6 / 3 |

- 6 개 모두 `tiling_model.py`(chooseConvChunks·chooseConvTiling 를 그대로 옮긴 계산기)의 예측과 tile_sizes 가 같다. VGG 9 종 표도 [1109_native_conv_algorithm.md](1109_native_conv_algorithm.md) §4.4 와 같다.
- 끝 주소가 합보다 큰 것(c42 60,416 > 52,736)은 뱅크 단위 배치 때문이다. 버퍼가 16 KB 경계(16,384 · 32,768 · 49,152)에서 시작한다. 그래도 64 KB 안에 든다.
- memtile 은 512 KB 중 최대 136 KB 만 쓴다.
- BD 최대 사용: memtile 24(8·4 열) / 36(7 열) 개로 한도 48 이하, 코어 6 개로 한도 16 이하.
- `conv_dma_check.py`(원소 단위 DMA 대조): c12 · c42 PASS. 출력 원소를 정확히 한 번씩 덮는다.

## 6. conv-decompose 와 비교

설정(`KernelDispatch.cpp:1058-1074`)은 다음과 같다.

| 단계 | (N, OH, OW, OC, KH, KW, IC) | 뜻 |
|---|---|---|
| 블록 | [1, 4, 8, 8, 0, 0, 0] | 출력 4 행 × 8 열 × 8 채널, IC 전부 |
| 코어 | [1, 1, 8, 8, 0, 0, 0] | 코어 하나 = 출력 1 행 → OH 4 행이 코어 4 개 |
| 패킹 | OC 8, IC 8 | npu4 bf16 matmul 명령 8 × 8 × 8 (`iree_aie_runtime.cc:1014-1016`) |
| 코어 안 | KH 1, KW 1, IC/8 1 씩 `scf.for` | L1 에 이미 있는 IC 위를 도는 계산 루프 |

N = 2 conv 두 개를 npu4 로 직접 컴파일했다(`log_shjo/partition/cd/`, 벡터화 끔).

| 입력 | 결과 | IR |
|---|---|---|
| N=2, 16² (입력 18²), IC 32 → OC 64 | 성공 | 타일은 열 0 뿐: shim (0,0), memtile (0,1), 코어 (0,2)–(0,5). 코어 버퍼 입력 [1][3][10][32] 1,920 B · 필터 [3][3][32][8] 4,608 B · 출력 256 B, 각각 × 2 |
| N=2, 32² (입력 34²), IC 512 → OC 512 | 실패: `'aie.memtile_dma' op could not find and assign a valid BD id` | `aie.dma_bd` 8,244 개(memtile 한도 48). 코어 버퍼 입력 30,720 B · 필터 73,728 B, 각각 × 2 → 약 209 KB (> 64 KB) |

- **공간**: 코어 4 개가 OH 4 행을 나눈다. 한 열뿐이다.
  - 설정이 처음부터 한 열을 겨냥한다(코드 주석 "Target one column of 4 cores").
  - conv-decompose 에서는 memtile 분할 패스가 돌지 않는다(`Passes.cpp:1153-1158`). 그래서 memtile 도 하나다.
- **시간**: 블록 forall (N, OH/4, OW/8, OC/8) 이 시간이다. 코어 안에서 KH × KW × IC/8 루프가 돈다.
- **IC**: 블록·코어 타일 모두 IC 를 자르지 않는다. 입력 행 3 개 × 10 열 × IC 와 필터 3 × 3 × IC × 8 이 통째로 L1 에 올라간다.
  - 이중 버퍼 L1 ≈ 408 · IC + 512 B 이므로 IC ≤ 156 에서만 들어간다(npu4 bf16).
  - VGG 의 IC 256·512 층은 L1 에서 막힌다. `conv` 파이프라인이 채널 루프를 코어 forall 밖에 둔 것이 이 벽을 없앤 핵심이다.
- **BD**: IC 512 의 BD 폭증은 9 월에 `dev-conv` 에서 고친 문제다(`855c3168`, 쓰는 쪽 repetitionCount 만 접기). 이 커밋은 `dev-conv`·`dev-GEMV` 에만 있고 `conv-npu` 에는 없다.
- **받는 형태**:
  - generic 은 반복자 7 개여야 한다. 배치 1 이 접힌 실제 모델의 conv 는 해당하지 않는다.
  - OW 가 8 의 배수여야 한다. OW 12 는 `memref.subview … dynamic shape` 로 실패했다. VGG 의 28·14·7 도 해당하지 않는다.
- 벡터화를 켜면 작은 conv 도 aievec 변환에서 bf16 `vector.contract` 가 거부되어 실패했다. 분할과는 무관한 단계다.

| | `conv` 파이프라인 | conv-decompose |
|---|---|---|
| 공간 | 청크 → 열(최대 8), OC 타일 → 행(4) | OH 4 행 → 코어 4 개(1 열) |
| 시간 | 블록 (OH·OW·OC) + IC 단계 | 블록 (N, OH/4, OW/8, OC/8) |
| 코어 안 | 타일 × 3 × 3 × icTile | 1 행 × 8 OW × 8 OC, KH·KW·IC/8 루프 |
| IC 와 L1 | 흘려 넣음 → VGG 전 층 L1 안 | 통째로 올림 → IC ≤ 156 |

## 7. 관찰과 한계 (정적 추론, NPU 미측정)

- **코어 사용률**: OH 28·14 층은 32 코어 중 28 개를 쓴다. 8 번째 열을 쓰려면 OW 도 나누는 2 차원 청크가 필요하다. OH 7 은 계산 행의 1/8 이 패딩이다.
- **채널 패딩**: conv1_1 은 IC 3 → 8 이라 MAC 이 2.7 배다(층 자체는 작다). 나머지는 12.5 % 이하(64 → 72)다.
- **L2 를 거의 쓰지 않는다**: memtile 은 512 KB 중 136 KB 이하만 쓴다. 블록 사이 재사용이 없어 conv1_2 는 83 KB 필터를 32 번(2.65 MB), 깊은 층은 입력을 8 번 다시 읽는다.
- **채널이 한계에 붙어 있다**: memtile S2MM 은 모든 층에서 6/6 이고, 7 열 층은 MM2S 도 6/6 이다. 스트림을 하나라도 더하는 변경(세 번째 입력, 부분합, 두 번째 출력)은 배치를 바꿔야 한다.
- **출력이 f32 다**: 배출량이 bf16 의 두 배이고, L1 에서 누산기와 epilogue 출력이 가장 큰 몫이다.
- **NCHW 출력은 받지 않는다**: conv 결과를 바로 NCHW 로 내보내는 모델(단일 층 ONNX 의 출력처럼)은 IREE 가 ReLU 와 전치를 한 epilogue 로 묶는다. 전략이 이를 거부한다: `native conv pipeline: its result is read by 'linalg.generic', which is not a unary elementwise epilogue with identity maps (a residual or bias add, or a transposed output, is not supported)`(`KernelDispatch.cpp:779-786`). VGG 는 conv 뒤가 conv·pooling(channels-last)이라 해당하지 않는다. 이 문서의 모델은 입출력을 NHWC 로 두어 피했다.
- 시간·수치는 NPU 에서 재지 않아 모른다. 위 트래픽은 IR 의 BD 크기로 셈한 바이트 수이고, 시간이 아니다.

## 8. 재현

모든 도구와 입력은 `log_shjo/partition/`(커밋하지 않음)에 있다.

| 파일 | 하는 일 |
|---|---|
| `gen_conv_onnx.py <out> H IC OC [B]` | Conv 3×3(pad 1, bias) + Relu 단일 층 ONNX. 입출력 NHWC (Transpose 쌍이 상쇄되게) |
| `compile_all.sh [모델…]` | `run-onnx.sh compile` 에 덤프 플래그를 `--extra-flag` 로 붙여 컴파일 |
| `tiling_model.py [H IC OC …]` | chooseConvChunks·chooseConvTiling 를 그대로 옮긴 표 계산기. IR 과 6/6 일치 |
| `buffers.py <compile.log> [col,row]` | amdaie-assign-buffer-addresses 덤프에서 memtile·코어 버퍼 |
| `cd/*.mlir` | conv-decompose 비교 입력 (N = 2) |

덤프 지점: `iree-amdaie-chunk-conv-dispatches`, `iree-amdaie-lowering-strategy`, `iree-amdaie-lower-executable-target`, `iree-amdaie-split-logical-objectfifos`, `iree-amdaie-assign-channels`, `iree-amdaie-assign-npu-dma-bd-ids`, `iree-amdaie-lower-to-aie`, `amdaie-assign-bd-ids`(before), `amdaie-assign-buffer-addresses`. 큰 상수는 `--mlir-elide-elementsattrs-if-larger=16 --mlir-elide-resource-strings-if-larger=16` 로 생략했다.

```bash
./log_shjo/partition/compile_all.sh pc_c42_28x512x512
```

```bash
python3 -I log_shjo/partition/tiling_model.py 28 512 512 14 512 512
```

conv-decompose 비교는 dev 컨테이너에서 `build/tools/iree-compile <cd/*.mlir> --iree-hal-target-device=amdxdna --iree-amdaie-target-device=npu4 --iree-amdaie-tile-pipeline=conv-decompose --iree-amdaie-lower-to-aie-pipeline=objectFifo --iree-amdaie-enable-vectorization-passes=false` 로 했다.
