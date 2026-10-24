# 3장치(CPU + NPU + GPU) 구현 명세

CLAUDE.md 1-4의 2(구현 명세) 형식의 주제별 명세다. 변경 장부는 `docs/CHANGES.md`.

- 첫 판: 2026-10-20 세션 1 챕터 02(로컬 세션 문서 `docs/2026-10-20/1_변경장부_명세_커밋준비/02_구현_명세/`, 팀 git 미포함). 이 폴더는 팀 git에 올리는 사본이다. 세션 문서를 가리키는 링크 2곳(S8·S9의 경계 버퍼 동기화 표)만 경로 글로 바꿨다.
- 문서 안의 `docs/2026-…` 경로는 로컬 세션 기록이다(팀 git에 없음).

| 문서 | 주제 | 구분 |
|---|---|---|
| S1 | `ElideAsyncCopies` 양방향 topology 판정 | 수정 Pass(IREE) |
| S2 | amdxdna 동기 할당 `placement.device` | 런타임 |
| S3 | Vulkan 남의 버퍼 BRIDGE 공통화 + 명시적 에러 | 런타임 |
| S4 | Vulkan 영구 매핑 + 디버그 로그 map 분리 | 런타임 |
| S5 | `AMDAIEAnnotateInputDevices` + 입력 자동 배치 | 신규 Pass + 도구 |
| S6 | (삭제) `respect-marks` | dev 통합 때 옵션과 함께 제거(디버그 전용) |
| S7 | `placement-map` | 수정 Pass(실험 옵션) |
| S8 | 전임자 `ResolveBufferOwner` + stream hook + 링크 표 | 인수인계 기준선 |
| S9 | 전임자 런타임 공유 경로 | 인수인계 기준선 |
| S10 | (삭제) ukernel bf16 반올림 | dev에 반영됨(`cff1e56e`, 반올림 모드 저장·복원 포함) |
