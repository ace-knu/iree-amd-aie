# Attention softmax NPU handoff (2026-09-17)

## 상태

attention-only head-split 모델에서 softmax를 포함한 attention 경로가 실제 NPU에서
컴파일·실행·정확도·결정성 검증까지 통과했다.

```text
NPU: Q, K, V projection / QKᵀ / i8 softmax / PV / output projection (7)
CPU: input quantize+pad / head broadcast / final dequant+bias (3)
```

Softmax는 더 이상 CPU dispatch가 아니다. `dequant -> softmax -> quant` 체인이
각 core의 `[3,32,32]` tile에서 `softmax_i8_96x32` 하나로 낮아진다.

## 검증

- NPU softmax tap vs ORT: corr `0.9992575`, maxdiff `3`, mean abs diff `0.087321`.
- 최종 attention output vs ORT: corr `0.9987321`, maxdiff `0.0474694`, mean abs diff
  `0.0060457`.
- tap 없는 원래 attention-only 모델도 tap 모델의 최종 output과 SHA-256이 동일하다.
- 5회 NPU 실행은 최종 output과 softmax tap 모두 각각 단일 SHA-256이다.

## 성능

동일 입력, 동일 debug build, `iree-benchmark-module`, 5 repetitions, NPU lock 사용:

| softmax 위치 | 평균 wall time | 중앙값 | CPU process time |
|---|---:|---:|---:|
| CPU baseline | 23.6 ms | 23.6 ms | 2.97 ms |
| NPU i8 softmax | 25.3 ms | 25.2 ms | 1.68 ms |

- NPU softmax는 CPU 작업 시간을 줄이지만 end-to-end wall time은 약 **7.2% 느리다**.
- 현재 `[12,32,32]` softmax에서는 별도 NPU dispatch의 launch/DMA/synchronization
  비용이 계산 이득보다 크다.

## 핵심 구현

1. `softmax_i8_96x32`/`softmax_i8_32x32` Peano ukernel: i8 입력·출력과 QDQ scale을
   kernel 내부 BF16 계산에 접는다.
2. strict QDQ-softmax-QDQ DAG matcher와 softmax ukernel enabled일 때만 NPU로 보내는
   affinity를 추가했다.
3. general-copy tile-and-fuse를 producer/root/elementwise-consumer chain에 맞게
   확장하고, chain 내부 edge가 L2/L1 promotion으로 끊기지 않게 했다.
4. `scf.forall` consumer fusion 후 남는 dead f32 root output을 제거했다.
5. softmax wrapper의 offset 타입을 `unsigned`에서 `int64_t`로 고쳤다. 공통
   `UKernelGenericOp` lowering은 memref offset을 i64로 호출하므로 32-bit wrapper는
   output pointer와 scale 인자의 ABI 위치를 어긋나게 해 softmax output을 모두 0으로
   만들었다.

## 남은 CPU dispatch와 해결 방향

### d0: input quantize + pad

CPU f32 model input을 i8 NPU input shape로 만드는 ingress boundary다. 일반적인 해결은
host pre-processing을 유지하거나, Q/K/V consumer의 input pack에 quantize+pad를 fuse하는
것이다. 후자는 API/input representation과 precision 정책을 함께 설계해야 한다.

### d1: 12-head broadcast

독립 NPU broadcast dispatch를 추가하는 것은 launch 비용 때문에 이득이 보장되지 않는다.
범용적 방향은 broadcast를 Q/K/V consumer tile-and-fuse 또는 DMA replication으로 표현해
중간 12벌 materialization을 없애는 것이다.

### d9: final dequant + bias

f32 model output을 요구하는 egress boundary다. 단순히 NPU로 보내면 scalar f32 연산과
별도 dispatch 비용이 생긴다. 필요할 때만 projection epilogue에 BF16/f32 dequant+bias를
지원하고, 정확도·출력 ABI·전송량을 함께 평가한다.

## 권장 다음 순서

1. 현 NPU7/CPU3 경로를 attention-only 기준선으로 보존한다.
2. d1 broadcast를 Q/K/V 쪽으로 fuse할 수 있는지 IR·DMA 비용부터 측정한다.
3. d0/d9는 모델 I/O boundary로 남겨도 되는지 제품 요구사항을 먼저 정한다.
4. softmax 성능을 더 개선하려면 standalone dispatch를 줄이는 QKᵀ→softmax→PV scheduling
   또는 launch/DMA amortization을 연구한다. 단순히 softmax만 NPU에 올리는 것은 현재
   성능상 손해다.

## 범위와 주의

- 이 결과는 all-zero mask를 상수 폴딩한 attention-only 모델이다. full BERT mask는
  input_ids에서 계산되므로 별도 검증이 필요하다.
- full BERT의 FC1 DMA/channel 문제는 이 작업 범위 밖이며 해결되지 않았다.
