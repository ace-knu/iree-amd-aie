# 이종 device 배치 (Heterogeneous device placement)

amd-aie 백엔드에서 여러 device(가속기 NPU·GPU + host CPU)에 dispatch를 나눠 배치하는 방식과, 그 구조를 어떻게 확장하는지 정리한다.
CPU + NPU, CPU + GPU, CPU + NPU + GPU 구성을 지원한다(3devices-f32 통합 후, 2026-10-25). 통합 경위는 `docs/CHANGES.md`의 "dev 통합 후 상태"를 본다.

## 개요

- **패스**: `AMDAIEAssignDeviceAffinities`
  (`compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEAssignDeviceAffinities.cpp`)
- **역할**: contraction/convolution을 **가속기**(amd-aie, vulkan-spirv)에, 그 외(layout transpose, demote가 만든
  f32→bf16 cast, 기타 elementwise)를 **host(CPU)**에 배치한다. Flow 단계 끝(dispatch outlining 후, Stream affinity
  solver 전)에서 각 `flow.dispatch`에 `stream.affinity`를 부여하고, cross-device 버퍼 해결을 위한 `stream.topology`를 주입한다.
- **훅**: `PluginRegistration.cpp`
  - `extendPreprocessingPassPipeline` — topology 조기 주입(host device global이 Flow 전에 DCE되는 것 방지) + demote.
  - `extendFlowTransformPassPipeline` — dispatch 형성 후 affinity 부여.
  - `extendStreamTransformPassPipeline` — `AMDAIEResolveBufferOwner`(공유 버퍼 소유자 결정), `AMDAIEAnnotateInputDevices`
    (입력별 장치를 reflection `amdaie.input_devices`에 기록). IREE fork의 stream 단계 plugin hook을 쓴다.

## device-capability 테이블 (단일 출처)

지원 device의 역할과 device 쌍의 topology capability를 **한 곳**에 두고, 분류와 topology 두 seam이 이 테이블을 참조한다.

| 조회 | 현재 값 |
|---|---|
| `deviceRole(backend)` → Host / Accelerator / Unknown | `llvm-cpu`=Host, `amd-aie`·`vulkan-spirv`=Accelerator, 그 외=Unknown(무시) |
| `linkFlags(srcBackend, dstBackend)` → `{unified_memory, transparent_access}` | 아래 링크 표. 표에 없는 쌍은 둘 다 `false`(보수적 staging) |

링크 `src → dst`의 `transparent_access`는 "src가 dst가 할당한 메모리를 복사 없이 접근한다"는 뜻이다. 두 방향은 따로 정한다.

| src → dst | transparent | 근거 |
|---|---|---|
| CPU → NPU | true | amdxdna BO는 host-local·device-visible이라 CPU dispatch가 host-map한다 |
| NPU → CPU | false | amdxdna HAL allocator에 `import_buffer`가 없어 NPU가 CPU heap 할당을 바인딩하지 못한다 |
| CPU → GPU | true | 이 장비(UMA)의 Vulkan dispatch 메모리는 DEVICE_LOCAL\|HOST_VISIBLE\|HOST_COHERENT라 CPU가 host-map한다. 외장 GPU면 false여야 한다 |
| GPU → CPU | false | Vulkan BRIDGE는 명령을 기록할 때 CPU 버퍼를 host 포인터로 import하는데, queue 순서 할당(transient)은 그때 아직 메모리가 없다. BRIDGE가 실행 시점에 import하게 되면 true로 되돌릴 수 있다 |
| GPU → NPU | true | amdxdna가 BO를 dma-buf fd로 export하고 Vulkan이 import한다(BRIDGE가 자동으로 처리) |
| NPU → GPU | false | amdxdna에 import가 없고 Vulkan allocator는 export하지 않는다 |

이 표가 공유 버퍼를 누가 소유할 수 있는지의 유일한 출처다. 행을 바꾸면 `ElideAsyncCopies`가 지우는 transfer도 달라지므로
동작 변경으로 보고 `tests/regression/`을 다시 돌린다.

> `transparent_access`는 장치쌍·방향에 따라 다른 하드웨어 속성이라(IREE `HALAttrs.td`의 예시가 쌍마다 다름) **무조건 true는
> 일반적으로 틀리다.** IREE는 topology를 자동 유도하지 않고 소비만 하므로(`ResolveTopologyQueries`/`ElideAsyncCopies`),
> 이 테이블이 알려진 값을 인코딩하고, 사용자가 `stream.topology`를 직접 주면 그걸 우선한다.

## 현재 동작

1. **분류** — 선언된 모든 device global(`--iree-hal-target-device` 옵션)의 backend를 `deviceRole`로 조회 →
   `accelerators` / `hosts` 리스트. 미등록 backend는 skip(배정·topology 어디에도 안 들어가 DCE).
2. **조건** — `host ≥ 1` 필수(가속기는 선택). 단일 host 구성도 동작.
3. **정책(round-robin)** — contraction/conv dispatch는 가속기에 선언 순서대로 번갈아 배정한다
   (`accelerators[accelIdx++ % N]`, `accelIdx`는 0에서 시작). 그 외는 `hosts.front()`. 가속기가 없으면 contraction도 host로.
4. **placement-map(실험)** — `--iree-amdaie-placement-map`이 비어 있지 않으면 규칙에 맞는 dispatch를 그 장치에 고정한다.
   맞는 규칙이 없는 dispatch는 3의 정책을 따른다(아래 "placement-map" 참고).
5. **topology** — 실제 배정된 device만 mesh하고 링크 플래그는 테이블에서 조회한다. 이 패스는 두 번 돈다: preprocessing에서는
   dispatch가 아직 없어 모든 후보를 mesh(가속기 global이 DCE되지 않게)하고, Flow에서 실제로 쓴 device로 줄인다. device가
   둘 이상일 때만 넣는다. 사용자 제공 `stream.topology`가 있으면 생성/덮어쓰지 않음.

## 장치 역할: lead, 기본 장치, 소유자

| 개념 | 정하는 곳 | 하는 일 |
|---|---|---|
| 기본 장치 | `--iree-hal-default-device` | affinity가 없는 op(host 코드 등)를 둔다 |
| lead | 런타임 `--device_lead_allocator`(기본: 첫 `--device`) | 툴링이 입력을 할당하고 출력을 받는 장치. 소유자를 정하지 않는다 |
| 소유자 | 컴파일러 `AMDAIEResolveBufferOwner` | 여러 장치가 쓰는 버퍼를 어느 장치가 할당할지 정한다 |

- `scripts/onnx/`는 CPU(`local-host`)를 첫 장치로 강제한다. CPU가 host 코드를 실행하므로 기본 장치이자 lead가 된다.
- **소유자 규칙**: 후보 X는 다른 모든 후보 Y에 대해 `(Y → X)`가 transparent일 때만 소유자가 될 수 있다. 자격자가 정확히
  하나면 그 장치로 정하고, 0개나 여럿이면 `#hal.device.optimal`을 그대로 두어 런타임이 고른다. 위 링크 표로는 CPU+NPU는
  NPU, CPU+GPU는 GPU, GPU+NPU는 NPU, 세 장치가 함께 쓰면 NPU가 소유한다.
- **입력 배치**: `AMDAIEAnnotateInputDevices`가 같은 규칙으로 입력별 장치를 reflection에 기록하고, `iree-run-module`이
  `--input_device` 없이 그 장치로 입력을 옮긴다.

## placement-map (실험)

기본값은 비어 있고 모델별 맵은 커밋하지 않는다. 배정을 바꿔 보는 실험에만 쓴다.

```bash
# 2번 dispatch는 NPU, 이름에 matmul이 들어간 dispatch는 GPU, 나머지는 CPU
--iree-amdaie-placement-map='d2=npu;~matmul=gpu;*=cpu'
# 파일로: 한 줄에 규칙 하나, '#' 주석
--iree-amdaie-placement-map=@map.txt
```

- 선택자: `dN`, `dN-dM`, `dA,dB`(실행 파일 이름의 `_dispatch_N` 번호), `~substr`(실행 파일·export 이름), `*`.
- 장치: `npu`/`gpu`/`cpu`(그 backend의 첫 device global) 또는 `@<device global>`. 처음 맞는 규칙이 이긴다.
- 플래그를 주면 dispatch별 배정 표를 stderr에 출력한다. 아무것에도 맞지 않는 번호 규칙은 오류다.
- `scripts/onnx/`에서는 `--extra-flag=--iree-amdaie-placement-map=...`로 넘긴다. 명세: `docs/specs/S7_placement_map.md`.

## 확장 가이드

| 상황 | 필요한 작업 |
|---|---|
| **같은 backend의 새 device** (예: npu4 외 다른 amd-aie 타겟) | 테이블 수정 불필요 — 이미 amd-aie로 분류됨. 옵션으로 타겟하면 그대로 동작. |
| **새 accelerator backend** (예: 다른 종류 NPU) | `deviceRole`에 role 추가 + `linkFlags`에 쌍 추가. **단, 테이블 수정만으로는 부족하다** — (1) 분류 술어(`executableIsContractionOrConv`)가 모든 accelerator에 공유되는 하나라 op 커버리지가 다른 두 accelerator를 구분할 수 없고, (2) affinity를 무조건 덮어쓰므로 두 번째 accelerator 플러그인이 자기 placement 패스를 등록하면 서로 덮어쓴다. 그리고 그 backend의 실제 codegen 지원이 있어야 한다 — codegen 없는 backend에 contraction을 배정하면 downstream 실패. 상세는 `AMDAIEAssignDeviceAffinities.cpp`의 테이블 주석 참고. |
| **다중 동종 device 실제 분배** (예: NPU 2개를 나눠 씀) | 테이블 수정 불필요 — round-robin이 선언 순서대로 번갈아 배정한다. 성능을 보고 나누려면 정책 seam(round-robin) 교체가 필요하다. |
| **새 GPU·외장 GPU** | 같은 vulkan-spirv면 role은 그대로. 외장 GPU처럼 dispatch 메모리가 host-visible이 아니면 링크 표의 CPU → GPU를 false로 바꿔야 한다. |

## 미래: 지원연산 + 성능 기반 배치 정책의 기반

각 가속기가 실행 가능한 연산이 다르고, 정책이 "각 device의 지원 연산 + 성능"을 종합해 배치를 결정하는 형태로 확장할 때,
이 테이블 + 정책 seam이 **그대로 기반**이 된다.

```
device-capability 테이블 (device 당)
  role         : Host / Accelerator            ← (현재)
  linkFlags    : 쌍별 topology 링크 capability   ← (현재)
  supportedOps : 이 device가 실행 가능한 op 집합   ← (추가)
  cost         : (device, opKind) → 성능/비용     ← (추가)

정책(현재 round-robin → 미래):
  각 dispatch(op kind, shape 등)에 대해
    1) supportedOps로 후보 device 필터
    2) cost로 최적 device 선택 (연산 성능 + cross-device 전송비용 종합)
```

설계상 이점:
- **role을 supportedOps에서 파생 가능** — "contraction/conv를 지원하면 accelerator, 범용 fallback이면 host"로 두면
  `supportedOps`가 `role` 컬럼을 흡수해 더 단일화된다.
- **전송비용까지 한 곳에서** — topology 링크 정보(`linkFlags`, transparent/staging)가 같은 테이블에 있으니, cost 정책이
  연산 성능뿐 아니라 cross-device 전송비용까지 종합해 비교할 수 있다.
- **정책 seam이 이미 분리됨** — 현재 round-robin 선택 로직 자리가 그대로 정책 교체 지점.
- **2단계 topology 이미 있음** — per-dispatch 정책은 dispatch를 봐야 결정되므로 preprocessing에서 후보 전체를 keep-alive
  mesh → Flow에서 실제 used-set으로 narrowing해야 한다. round-robin을 위해 이미 이렇게 되어 있다
  (`AMDAIEAssignDeviceAffinities.cpp` topology 주석).

정직한 단서(규모 확장 시):
- **supportedOps 출처**: 소수 backend는 테이블 하드코딩으로 충분하나, 규모가 커지면 각 backend의 실제 codegen 능력을
  **질의(registry)** 하는 편이 이상적(backend가 자기가 lower 가능한 op을 앎).
- **cost 모델**: 실제 비용 모델(shape·메모리·전송 고려)은 그 자체로 큰 작업. 단순 힌트(상대 처리량)에서 점진 확장 가능.
- **scope**: 진짜 다중-가속기 성능 정책은 여러 backend codegen을 아우르므로 IREE-core/멀티-플러그인 성격도 있다. 다만
  "테이블 + 정책 seam" 패턴이 그 로컬 기반으로 맞고, core 메커니즘과 정렬시키기 좋다.

## 참고

- 테이블: `Transforms/Utils/AMDAIEDevicePlacementUtils.{h,cpp}` (`deviceRole`/`linkFlags`/`getFirstDeviceWithRole`).
- 소유자·입력 장치: `Transforms/AMDAIEResolveBufferOwner.cpp`, `Transforms/AMDAIEAnnotateInputDevices.cpp`
  (명세 `docs/specs/S8_*`, `S5_*`).
- 패스: `Transforms/AMDAIEAssignDeviceAffinities.cpp` (`runOnOperation`). 테이블을 읽는 다른 소비자는
  `Transforms/AMDAIEPadContractionDispatches.cpp`의 `getHostAffinity`.
- 훅: `PluginRegistration.cpp` (`extendPreprocessingPassPipeline`/`extendFlowTransformPassPipeline`/`extendStreamTransformPassPipeline`).
- HAL topology attr: IREE `compiler/src/iree/compiler/Dialect/HAL/IR/HALAttrs.td` (`DeviceTopologyAttr`/`DeviceLinkAttr`),
  소비처 `Dialect/HAL/Transforms/ResolveTopologyQueries.cpp`, `Dialect/Stream/Transforms/ElideAsyncCopies.cpp`.
- 실행: `scripts/onnx/README.md`(`--devices`), 3장치 회귀 `tests/regression/README.md`.
- 실행 레시피(2-장치 npu+cpu): `docs/2026-07-06_env_setup/USER_GUIDE.md`.
