# State 기반 워크플로우 동작 순서 가이드

## 개요

이 프로젝트는 4가지 상태(`start`, `write`, `evaluation`, `end`)를 통해 Triton 커널 개발 워크플로우를 관리합니다. 각 상태에서 사용 가능한 도구와 상태 전환 조건을 명확히 이해하는 것이 중요합니다.

## 상태 다이어그램

```
┌─────────┐
│  START  │  (정보 수집 단계)
└────┬────┘
     │ (모든 정보 수집 도구 호출 완료)
     ▼
┌─────────┐
│  WRITE  │  (커널 코드 작성 단계)
└────┬────┘
     │ (write_kernel_code 호출)
     ▼
┌─────────────┐
│ EVALUATION  │  (검증 및 평가 단계)
└─────┬───────┘
      │
      ├─→ (검증 실패 또는 최소 write 미충족)
      │   └─→ WRITE (재작성)
      │
      └─→ (검증 통과 + 최소 3번 write 완료)
          └─→ END (완료)
```

## 상태별 상세 설명

### 1. START 상태 (정보 수집 단계)

**목적**: Triton 커널 개발에 필요한 정보를 수집합니다.

**사용 가능한 도구**:
- `set_kernel_name(name)` - 커널 이름 설정 (로그 파일 생성)
- `get_overview()` - 전체 프로세스 개요
- `get_torch_op_info(op_name)` - PyTorch 연산 정보
- `get_triton_syntax()` - Triton 문법 레퍼런스
- `get_kernel_template(pattern)` - 커널 템플릿
- `check_gpu_status()` - GPU 상태 확인
- `get_current_status()` - 현재 상태 확인 (모든 상태에서 사용 가능)
- `check_context_usage()` - Context 사용량 확인 (모든 상태에서 사용 가능)

**상태 전환 조건**:
- 다음 4가지 정보 수집 도구를 모두 호출하면 **자동으로 `write` 상태로 전환**:
  1. `get_overview()`
  2. `get_torch_op_info()`
  3. `get_triton_syntax()`
  4. `check_gpu_status()`

**예시 워크플로우**:
```
1. set_kernel_name("add")           # 커널 이름 설정
2. get_overview()                   # 전체 프로세스 파악
3. get_torch_op_info("vector_add")  # 연산 정보 확인
4. get_triton_syntax()              # Triton 문법 참고
5. check_gpu_status()               # GPU 확인
   → 자동 전환: start → write
```

---

### 2. WRITE 상태 (커널 코드 작성 단계)

**목적**: Triton 커널 코드를 작성하고 저장합니다.

**사용 가능한 도구**:
- `write_test_code(test_code)` - 테스트 코드 작성 (선택사항, 첫 write 전에 권장)
- `write_kernel_code(code)` - 커널 코드 작성 및 저장
- `run_triton_kernel(test_input_code, entry_function)` - 커널 실행 테스트
- `validate_correctness(reference_code, test_input_code)` - 정확성 검증
- `get_current_status()` - 현재 상태 확인
- `check_context_usage()` - Context 사용량 확인

**상태 전환 조건**:
- `write_kernel_code()` 호출 시 **자동으로 `evaluation` 상태로 전환**
- 커널 버전이 자동으로 증가 (v1, v2, v3, ...)

**예시 워크플로우**:
```
1. write_test_code(test_code)      # 테스트 코드 작성 (선택)
2. write_kernel_code(code)          # 커널 코드 작성
   → 자동 전환: write → evaluation
   → 커널 파일 저장: triton_add_kernel_v1.py
```

---

### 3. EVALUATION 상태 (검증 및 평가 단계)

**목적**: 작성한 커널의 정확성과 성능을 검증합니다.

**사용 가능한 도구**:
- `run_triton_kernel(test_input_code, entry_function)` - 커널 실행
- `validate_correctness(reference_code, test_input_code)` - 정확성 검증
- `measure_kernel_time(test_input_code, warmup, rep)` - 실행 시간 측정
- `benchmark_kernel(test_input_code, reference_code, warmup, rep)` - 성능 벤치마크
- `force_transition_to_write()` - 수동으로 write 상태로 전환 (추가 최적화)
- `get_current_status()` - 현재 상태 확인
- `check_context_usage()` - Context 사용량 확인

**상태 전환 조건**:

1. **검증 실패 시**:
   - `validate_correctness()` 호출 후 검증 실패
   - **자동으로 `write` 상태로 전환** (코드 수정 필요)

2. **검증 통과 + 최소 write 미충족**:
   - `validate_correctness()` 호출 후 검증 통과
   - 하지만 `write_count < 3` (최소 3번 write 필요)
   - **자동으로 `write` 상태로 전환** (추가 최적화 권장)

3. **검증 통과 + 최소 write 충족**:
   - `validate_correctness()` 호출 후 검증 통과
   - `write_count >= 3` (최소 3번 write 완료)
   - **자동으로 `end` 상태로 전환** (완료)

4. **수동 전환**:
   - `force_transition_to_write()` 호출
   - **수동으로 `write` 상태로 전환** (추가 최적화 원할 때)

**예시 워크플로우**:
```
1. run_triton_kernel(test_input_code)           # 실행 테스트
2. validate_correctness(reference_code, ...)    # 정확성 검증

   # 시나리오 A: 검증 실패
   → 자동 전환: evaluation → write (재작성 필요)

   # 시나리오 B: 검증 통과, write_count < 3
   → 자동 전환: evaluation → write (추가 최적화)

   # 시나리오 C: 검증 통과, write_count >= 3
   → 자동 전환: evaluation → end (완료)
```

---

### 4. END 상태 (완료 단계)

**목적**: 최고 성능 커널을 선택하고 세션을 종료합니다.

**사용 가능한 도구**:
- `get_best_kernel()` - 최고 성능 커널 반환 및 로그 완료
- `set_kernel_name(name)` - 새 커널 시작 (자동으로 start 상태로 전환)
- `reset_session()` - 현재 세션 리셋 (start 상태로 전환)
- `get_current_status()` - 현재 상태 확인
- `check_context_usage()` - Context 사용량 확인

**상태 전환 조건**:
- `END`는 **종료 상태**이므로 더 이상 다른 상태로 자동 전환되지 않습니다
- `set_kernel_name()` 호출 시 **새 세션 시작** (start 상태로 전환)
- `reset_session()` 호출 시 **현재 세션 리셋** (start 상태로 전환)

**예시 워크플로우**:
```
1. get_best_kernel()              # 최고 성능 커널 반환
   → 로그 파일 완료: triton_add_log.md
   → 최고 성능 버전 정보 반환
```

---

## 전체 워크플로우 예시

### 정상적인 완료 시나리오

```
[START]
1. set_kernel_name("add")
2. get_overview()
3. get_torch_op_info("vector_add")
4. get_triton_syntax()
5. check_gpu_status()
   → 자동 전환: start → write

[WRITE #1]
6. write_kernel_code(code_v1)
   → 자동 전환: write → evaluation
   → 저장: triton_add_kernel_v1.py

[EVALUATION #1]
7. run_triton_kernel(test_input_code)
8. validate_correctness(reference_code, test_input_code)
   → 검증 통과, write_count=1 < 3
   → 자동 전환: evaluation → write

[WRITE #2]
9. write_kernel_code(code_v2)  # 최적화
   → 자동 전환: write → evaluation
   → 저장: triton_add_kernel_v2.py

[EVALUATION #2]
10. validate_correctness(...)
    → 검증 통과, write_count=2 < 3
    → 자동 전환: evaluation → write

[WRITE #3]
11. write_kernel_code(code_v3)  # 추가 최적화
    → 자동 전환: write → evaluation
    → 저장: triton_add_kernel_v3.py

[EVALUATION #3]
12. validate_correctness(...)
    → 검증 통과, write_count=3 >= 3
    → 자동 전환: evaluation → end

[END]
13. get_best_kernel()
    → 최고 성능 커널 반환 (v2가 가장 빠른 경우)
    → 로그 파일 완료
```

### 검증 실패 시나리오

```
[EVALUATION]
1. validate_correctness(...)
   → 검증 실패
   → 자동 전환: evaluation → write

[WRITE]
2. write_kernel_code(code_fixed)  # 버그 수정
   → 자동 전환: write → evaluation

[EVALUATION]
3. validate_correctness(...)
   → 검증 통과 또는 실패에 따라 계속 반복
```

---

## 상태 전환 규칙 요약

| 현재 상태 | 다음 상태 | 전환 조건 |
|----------|----------|----------|
| `start` | `write` | 모든 정보 수집 도구 호출 완료 (자동) |
| `write` | `evaluation` | `write_kernel_code()` 호출 (자동) |
| `evaluation` | `write` | 검증 실패 또는 최소 write 미충족 (자동) |
| `evaluation` | `write` | `force_transition_to_write()` 호출 (수동) |
| `evaluation` | `end` | 검증 통과 + `write_count >= 3` (자동) |
| `end` | `start` | `set_kernel_name()` 또는 `reset_session()` 호출 |

---

## 중요 규칙

### 1. 최소 3번 Write 제약

- `evaluation` → `end` 전환을 위해서는 **최소 3번의 `write_kernel_code()` 호출**이 필요합니다
- 검증이 통과했더라도 `write_count < 3`이면 자동으로 `write` 상태로 되돌아갑니다

### 2. 상태별 도구 접근 제어

- 각 도구는 특정 상태에서만 사용 가능합니다
- 잘못된 상태에서 도구를 호출하면 에러 메시지가 반환됩니다

### 3. 자동 상태 전환

- 대부분의 상태 전환은 **자동**으로 이루어집니다
- 수동 전환은 `force_transition_to_write()`만 가능합니다

### 4. 로깅

- 모든 도구 호출은 자동으로 로그에 기록됩니다
- JSON 로그: `{session_id}_{timestamp}.jsonl`
- 마크다운 로그: `triton_{kernel_name}_log.md`

---

## 문제 해결

### Q: "이 도구는 현재 상태에서 사용할 수 없습니다" 에러

**원인**: 현재 상태에서 해당 도구를 사용할 수 없음

**해결**:
- `get_current_status()`로 현재 상태 확인
- 해당 도구가 사용 가능한 상태로 전환 필요

### Q: evaluation에서 end로 전환되지 않음

**원인**: 최소 3번 write 조건 미충족

**해결**:
- `write_kernel_code()`를 최소 3번 호출해야 함
- 현재 `write_count` 확인 후 추가 write 필요

### Q: 검증은 통과했는데 write로 되돌아감

**원인**: 최소 3번 write 조건 미충족

**해결**:
- 정상 동작입니다. 최소 3번 write를 완료하면 end로 전환됩니다.

---

## 참고

- 상태 관리 코드: `mcp_for_triton_kernel/state.py`
- 도구 구현: `mcp_for_triton_kernel/tools/`
- 로그 파일: `mcp_for_triton_kernel/log/`
