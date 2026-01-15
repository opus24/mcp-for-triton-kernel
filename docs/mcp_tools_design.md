# MCP Tools 설계 문서

## 개요

이 문서는 Triton 커널 개발을 위한 MCP (Model Context Protocol) 서버의 도구 설계를 정의합니다. 상태 기반 워크플로우를 통해 체계적인 커널 개발 프로세스를 지원합니다.

## 디렉토리 구조

| 디렉토리 | 용도 |
|---------|------|
| `mcp_for_triton_kernel/log/` | 로그 파일 저장 (JSON, 마크다운) |
| `mcp_for_triton_kernel/kernel/` | Triton 커널 코드 저장 |
| `mcp_for_triton_kernel/tests/` | 테스트 코드 저장 |

### 파일 명명 규칙

- **커널 파일**: `mcp_for_triton_kernel/kernel/triton_{name}_kernel_v{version}.py`
- **JSON 로그**: `mcp_for_triton_kernel/log/{session_id}_{timestamp}.jsonl`
- **마크다운 로그**: `mcp_for_triton_kernel/log/triton_{name}_log.md` (자동 생성)

**참고**: 모든 생성 파일은 `mcp_for_triton_kernel/` 패키지 디렉토리 내부에 저장됩니다.

## 상태 (Status) 시스템

### 상태 정의

시스템은 다음 4가지 상태로 구성됩니다:

1. **`start`**: 초기 상태. 정보 수집 단계
2. **`write`**: 코드 작성 단계
3. **`evaluation`**: 코드 검증 및 평가 단계
4. **`end`**: 최종 완료 상태

### 상태 전환 규칙

```
start → write      : 필요한 정보를 모두 수집했을 때
write → evaluation : 커널 코드 작성이 완료되었을 때
evaluation → end   : 테스트 통과 + 최소 3번의 write 완료 조건 만족
evaluation → write : 테스트 실패 또는 추가 최적화 필요
```

**중요 제약사항:**
- `evaluation` 상태에서 테스트를 통과했더라도, 최소 3번의 `write`가 완료되어야 `end`로 전환 가능
- 이는 여러 버전의 커널을 시도하고 최적의 성능을 찾기 위한 설계

## 도구 (Tools) 분류 및 상태 제약

### 1. 정보 수집 도구 (Information Tools)

**시작 가능 상태:** `start`

#### `get_overview`
- **설명**: Triton 커널 개발의 전체 프로세스와 기본 구조 설명
- **상태 변경**: 없음
- **로그**: 자동 기록

#### `get_triton_syntax`
- **설명**: Triton 문법, tl 함수들, 제약사항 레퍼런스 제공
- **상태 변경**: 없음
- **로그**: 자동 기록

#### `get_torch_op_info(op_name: Optional[str] = None)`
- **설명**: PyTorch 연산 정보 조회
- **상태 변경**: 없음
- **로그**: 자동 기록

#### `get_kernel_template(pattern: str = "elementwise")`
- **설명**: 커널 템플릿 제공 (elementwise, reduction, matmul, fused)
- **상태 변경**: 없음
- **로그**: 자동 기록

#### `check_gpu_status`
- **설명**: GPU 가용성 및 상태 확인
- **상태 변경**: 없음
- **로그**: 자동 기록

**상태 전환 조건:**
- `start` → `write`: 다음 정보가 모두 수집되었을 때
  - `get_overview` 호출 완료
  - `get_torch_op_info` 호출 완료 (변환할 연산 정보 확인)
  - `get_triton_syntax` 호출 완료
  - `check_gpu_status` 호출 완료

### 2. 코드 작성 도구 (Writing Tools)

**시작 가능 상태:** `write`

#### `set_kernel_name(name: str)`
- **설명**: 커널 이름 설정 (로그 파일명, 커널 파일명에 사용)
- **상태 변경**: 없음
- **로그**: 마크다운 로그 파일 자동 생성 (`log/triton_{name}_log.md`)

#### `write_kernel_code(code: str)`
- **설명**: Triton 커널 코드 작성 및 저장
- **상태 변경**: `write` → `evaluation` (코드 작성 완료 시)
- **로그**: 자동 기록 (코드 버전, 작성 시간 포함)
- **저장 위치**: `kernel/triton_{name}_kernel_v{version}.py`
- **기능**:
  - 코드 버전 관리 (version 번호 자동 증가)
  - 코드 저장 및 이력 관리
  - 문법 검사 수행
  - 성능 최적화 팁 제공

**성능 최적화 팁 (자동 출력):**
- BLOCK_SIZE 튜닝 (64, 128, 256, 512, 1024)
- 메모리 접근 최적화 (Coalesced access)
- 연산 융합 (Fusion)
- @triton.autotune 사용법

**상태 전환 조건:**
- `write` → `evaluation`: `write_kernel_code` 호출 완료 시

### 3. 실행 및 검증 도구 (Execution & Validation Tools)

**시작 가능 상태:** `write`, `evaluation`

#### `run_triton_kernel(code: str, test_input_code: str, entry_function: str = "solve")`
- **설명**: Triton 커널 실행 테스트
- **상태 변경**: 없음
- **로그**: 자동 기록 (실행 시간, 성공/실패 여부 포함)

#### `validate_correctness(kernel_code: str, reference_code: str, test_input_code: str, rtol: float = 1e-5, atol: float = 1e-8)`
- **설명**: 커널 정확성 검증
- **상태 변경**: 
  - `evaluation` → `end` (검증 통과 + 최소 3번 write 완료)
  - `evaluation` → `write` (검증 실패)
- **로그**: 자동 기록 (검증 결과, 차이 통계 포함)

#### `measure_kernel_time(kernel_code: str, test_input_code: str, warmup: int = 25, rep: int = 100)`
- **설명**: Triton 커널 실행 시간 측정 (새로 추가되는 도구)
- **상태 변경**: 없음
- **로그**: 자동 기록 (측정 시간, 통계 정보 포함)
- **기능**:
  - 워밍업 실행 후 실제 측정
  - 평균, 표준편차, 최소/최대 시간 반환
  - 성능 이력 저장

#### `benchmark_kernel(kernel_code: str, test_input_code: str, reference_code: Optional[str] = None, warmup: int = 25, rep: int = 100)`
- **설명**: 커널 성능 벤치마크 (참조 구현과 비교)
- **상태 변경**: 없음
- **로그**: 자동 기록 (벤치마크 결과, 비교 정보 포함)

**상태 전환 조건:**
- `evaluation` → `end`: 
  - `validate_correctness` 통과
  - `write_kernel_code` 호출 횟수 >= 3
- `evaluation` → `write`:
  - `validate_correctness` 실패
  - 또는 추가 최적화가 필요한 경우

### 4. 완료 도구 (Completion Tools)

**시작 가능 상태:** `end`

#### `get_best_kernel()`
- **설명**: 가장 빠른 커널 정보 반환 및 종료
- **상태 변경**: 없음 (종료 상태)
- **로그**: 자동 기록
- **기능**:
  - 모든 버전의 커널 중 가장 빠른 버전 선택
  - 성능 비교 결과 제공
  - 최종 커널 코드 반환

## 로깅 시스템

### 로그 요구사항

모든 도구는 호출될 때마다 자동으로 다음 정보를 로그에 기록해야 합니다:

1. **기본 정보**
   - 타임스탬프
   - 도구 이름
   - 현재 상태
   - 호출 인자 (민감 정보 제외)

2. **실행 결과**
   - 성공/실패 여부
   - 실행 시간
   - 에러 메시지 (실패 시)

3. **상태 변경**
   - 이전 상태
   - 새 상태
   - 상태 변경 이유

4. **커널 관련 정보** (해당되는 경우)
   - 커널 버전
   - 성능 측정 결과
   - 검증 결과

### 로그 저장 형식

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "tool": "write_kernel_code",
  "status_before": "write",
  "status_after": "evaluation",
  "args": {
    "version": 1
  },
  "result": {
    "success": true,
    "execution_time_ms": 123.45
  },
  "kernel_info": {
    "version": 1,
    "code_length": 1024
  }
}
```

## 워크플로우 예시

### 정상적인 워크플로우

```
1. [start] get_overview() → 정보 수집
2. [start] get_torch_op_info("softmax") → 연산 정보 확인
3. [start] get_triton_syntax() → 문법 참고
4. [start] check_gpu_status() → GPU 확인
5. [start → write] 상태 전환 (정보 수집 완료)

6. [write] write_kernel_code(code, version=1) → 코드 작성
7. [write → evaluation] 상태 전환

8. [evaluation] run_triton_kernel() → 실행 테스트
9. [evaluation] validate_correctness() → 정확성 검증
   - 실패 → [evaluation → write] 상태 전환
   - 통과 → 계속

10. [evaluation] measure_kernel_time() → 시간 측정
11. [evaluation] benchmark_kernel() → 성능 벤치마크

12. [evaluation → write] 상태 전환 (최소 3번 write 조건 미충족)

13. [write] write_kernel_code(code, version=2) → 최적화된 코드 작성
14. [write → evaluation] 상태 전환

15. [evaluation] validate_correctness() → 재검증
16. [evaluation] measure_kernel_time() → 재측정

17. [evaluation → write] 상태 전환 (최소 3번 write 조건 미충족)

18. [write] write_kernel_code(code, version=3) → 추가 최적화
19. [write → evaluation] 상태 전환

20. [evaluation] validate_correctness() → 재검증 (통과)
21. [evaluation] measure_kernel_time() → 재측정
22. [evaluation → end] 상태 전환 (조건 만족: 검증 통과 + 3번 write 완료)

23. [end] get_best_kernel() → 최고 성능 커널 반환 및 종료
```

### 실패 시나리오

```
1-7. (위와 동일)

8. [evaluation] validate_correctness() → 검증 실패
9. [evaluation → write] 상태 전환 (검증 실패로 인한 재작성)

10. [write] write_kernel_code(code, version=2) → 수정된 코드 작성
11. [write → evaluation] 상태 전환

12. [evaluation] validate_correctness() → 재검증
    - 계속 실패하면 write ↔ evaluation 반복
    - 통과하면 위의 정상 워크플로우 계속
```

## 구현 고려사항

### 상태 관리

- 상태는 세션별로 관리되어야 함
- 상태 전환은 도구 내부에서 자동으로 수행
- 상태 전환 조건을 명확히 검증

### 버전 관리

- `write_kernel_code` 호출 시마다 버전 자동 증가
- 각 버전의 코드, 성능 측정 결과, 검증 결과 저장
- `end` 상태에서 모든 버전 비교하여 최고 성능 선택

### 최소 3번 Write 제약

- `write_kernel_code` 호출 횟수를 추적
- `evaluation` → `end` 전환 시 이 조건 확인
- 검증 통과했더라도 3번 미만이면 `write`로 되돌림

### 로깅 구현

- 모든 도구에 공통 로깅 데코레이터 적용
- 로그 파일은 `mcp_for_triton_kernel/log/` 디렉토리에 저장
- 파일명 형식: `{session_id}_{timestamp}.jsonl`
- 마크다운 로그는 `mcp_for_triton_kernel/log/triton_{name}_log.md` 형식으로 자동 생성

### 새로운 도구: `measure_kernel_time`

- `benchmark_kernel`과 유사하지만 참조 구현 비교 없이 순수 시간 측정
- 더 가벼운 성능 측정이 필요할 때 사용
- 측정 결과는 버전별로 저장하여 비교 가능하도록

## 도구별 상태 제약 요약표

| 도구 | 시작 가능 상태 | 상태 변경 |
|------|---------------|----------|
| `get_overview` | `start` | 없음 |
| `get_triton_syntax` | `start` | 없음 |
| `get_torch_op_info` | `start` | 없음 |
| `get_kernel_template` | `start` | 없음 |
| `check_gpu_status` | `start` | 없음 |
| `write_kernel_code` | `write` | `write` → `evaluation` |
| `run_triton_kernel` | `write`, `evaluation` | 없음 |
| `validate_correctness` | `write`, `evaluation` | `evaluation` → `end` (통과+조건) 또는 `evaluation` → `write` (실패) |
| `measure_kernel_time` | `write`, `evaluation` | 없음 |
| `benchmark_kernel` | `write`, `evaluation` | 없음 |
| `get_best_kernel` | `end` | 없음 (종료) |

## 확장 가능성

이 설계는 향후 다음 기능 추가를 고려할 수 있습니다:

1. **상태 롤백**: 이전 상태로 되돌리기
2. **병렬 평가**: 여러 버전의 커널을 동시에 평가
3. **자동 최적화**: 성능 분석 기반 자동 코드 개선 제안
4. **커뮤니티 공유**: 검증된 커널을 커뮤니티에 공유

