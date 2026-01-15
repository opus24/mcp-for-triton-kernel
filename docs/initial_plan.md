# MCP for Triton Kernel - 개발 플랜

> **프로젝트 개발 계획 문서**

---

## 프로젝트 목표

PyTorch 연산을 Triton GPU 커널로 변환하는 작업을 지원하는 MCP (Model Context Protocol) 서버 개발

---

## 주요 기능 요구사항

### 1. 상태 기반 워크플로우 시스템
- [x] 4가지 상태 정의 (start, write, evaluation, end)
- [x] 상태 전환 규칙 구현
- [x] 최소 3번 write 제약 조건
- [x] 상태별 도구 접근 제어

### 2. 디렉토리 구조 관리
- [x] log/ 디렉토리 (JSON, 마크다운 로그)
- [x] kernel/ 디렉토리 (커널 코드 파일)
- [x] tests/ 디렉토리 (테스트 파일)
- [x] mcp_for_triton_kernel/ 패키지 내부에 위치

### 3. 로깅 시스템
- [x] JSON 로그 (구조화된 로그)
- [x] 마크다운 로그 (인간이 읽기 쉬운 개발 로그)
- [x] 자동 로그 생성
- [x] 커널 버전별 추적

### 4. Context 관리
- [x] Context 사용량 추적
- [x] 70% 초과 시 자동 요약 생성
- [x] docs/summarization.md 자동 저장
- [x] 새 세션 시작 안내

### 5. Triton 제약 해결
- [x] 파일 기반 실행 방식
- [x] 임시 파일 생성 및 import
- [x] @jit 데코레이터 제약 해결

### 6. 성능 최적화 가이드
- [x] write_kernel_code()에 최적화 팁 포함
- [x] BLOCK_SIZE 튜닝 가이드
- [x] 메모리 접근 최적화 팁
- [x] Autotune 사용 예시

---

## MCP 도구 목록

### 정보 수집 도구 (start 상태)
- [x] `get_overview()` - 전체 프로세스 안내
- [x] `get_triton_syntax()` - Triton 문법 레퍼런스
- [x] `get_torch_op_info(op_name)` - PyTorch 연산 정보
- [x] `get_kernel_template(pattern)` - 커널 템플릿
- [x] `check_gpu_status()` - GPU 상태 확인

### 워크플로우 도구
- [x] `get_current_status()` - 현재 상태 확인
- [x] `check_context_usage()` - Context 사용량 확인
- [x] `set_kernel_name(name)` - 커널 이름 설정
- [x] `write_kernel_code(code)` - 커널 코드 작성 및 저장
- [x] `force_transition_to_write()` - 상태 강제 전환

### 실행/검증 도구 (write, evaluation 상태)
- [x] `run_triton_kernel(test_input_code)` - 커널 실행
- [x] `validate_correctness(reference_code, test_input_code)` - 정확성 검증
- [x] `measure_kernel_time(test_input_code)` - 시간 측정
- [x] `benchmark_kernel(test_input_code, reference_code)` - 성능 벤치마크

### 완료 도구 (end 상태)
- [x] `get_best_kernel()` - 최고 성능 커널 반환

---

## 파일 구조

```
mcp-for-triton-kernel/
├── docs/
│   ├── plan.md                    # 이 문서
│   ├── mcp_tools_design.md        # 설계 문서
│   └── summarization.md           # 자동 생성 요약
├── mcp_for_triton_kernel/
│   ├── log/                       # 로그 파일
│   ├── kernel/                    # 커널 코드
│   ├── tests/                     # 테스트 파일
│   ├── state.py                   # 상태 관리
│   ├── server.py                  # MCP 서버
│   ├── tools/                     # MCP 도구들
│   ├── knowledge/                 # 지식 베이스
│   └── utils/                     # 유틸리티
│       ├── runner.py              # 커널 실행
│       └── context_manager.py    # Context 관리
└── README.md
```

---

## 개발 진행 순서

### Phase 1: 기본 구조 ✅
- [x] MCP 서버 기본 구조
- [x] 정보 수집 도구
- [x] 실행/검증 도구

### Phase 2: 상태 관리 시스템 ✅
- [x] StateManager 클래스
- [x] 상태 전환 로직
- [x] 커널 버전 관리

### Phase 3: 로깅 시스템 ✅
- [x] JSON 로그
- [x] 마크다운 로그
- [x] 자동 로그 생성

### Phase 4: 디렉토리 구조 ✅
- [x] log/, kernel/, tests/ 디렉토리
- [x] 패키지 내부 위치
- [x] 자동 디렉토리 생성

### Phase 5: Triton 제약 해결 ✅
- [x] 파일 기반 실행
- [x] 임시 모듈 관리
- [x] 자동 정리

### Phase 6: Context 관리 ✅
- [x] 사용량 추적
- [x] 자동 요약 생성
- [x] 새 세션 안내

### Phase 7: 성능 최적화 가이드 ✅
- [x] 최적화 팁 통합
- [x] BLOCK_SIZE 가이드
- [x] Autotune 예시

---

## 기술 스택

- **Python**: >= 3.10
- **PyTorch**: >= 2.0.0
- **Triton**: >= 2.1.0
- **FastMCP**: >= 2.14.0
- **CUDA**: GPU 실행 시 필수

---

