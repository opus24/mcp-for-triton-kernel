# MCP for Triton Kernel - 프로젝트 핸드오프

> **이 문서는 다음 에이전트를 위한 컨텍스트 전달용입니다.**

---

## 1. 프로젝트 개요

PyTorch 연산(예: `vector_add`, `softmax`)을 Triton GPU 커널로 변환하는 작업을 지원하는 **MCP 서버**입니다. Cursor IDE에서 LLM이 이 도구들을 호출하여 커널을 개발할 수 있습니다.

### 현재 상태

| 항목 | 상태 |
|------|------|
| 코드 구현 | ✅ 완료 |
| GPU 테스트 | ❌ 미진행 (GPU 없는 서버에서 개발) |
| Cursor 연동 테스트 | ❌ 미진행 |

---

## 2. 프로젝트 구조

```
/root/mcp-for-triton-kernel/
├── pyproject.toml              # fastmcp, torch, triton 의존성
├── README.md                   # 사용 가이드
├── mcp_config_example.json     # Cursor MCP 설정 예시
├── docs/
│   └── handoff.md              # 이 문서
└── mcp_for_triton_kernel/
    ├── server.py               # MCP 서버 엔트리포인트 (FastMCP 사용)
    ├── __main__.py             # python -m 실행 지원
    ├── __init__.py
    ├── tools/
    │   ├── __init__.py
    │   ├── info.py             # 정보 제공 도구 4개
    │   └── execution.py        # 실행/검증 도구 4개
    ├── knowledge/
    │   ├── __init__.py
    │   ├── overview.md         # Triton 개발 가이드
    │   ├── triton_syntax.md    # tl 함수, 문법 레퍼런스
    │   └── torch_ops.json      # 12개 torch ops 정보
    └── utils/
        ├── __init__.py
        └── runner.py           # TritonRunner 클래스 (커널 동적 실행)
```

---

## 3. 구현된 MCP Tools (8개)

### 정보 제공 도구 (tools/info.py)

| Tool | 설명 |
|------|------|
| `get_overview()` | Triton 커널 개발 전체 프로세스 안내 |
| `get_triton_syntax()` | Triton 문법, tl 함수 레퍼런스 |
| `get_torch_op_info(op_name)` | torch ops 정보 조회 (12개 연산) |
| `get_kernel_template(pattern)` | 패턴별 템플릿 (elementwise/reduction/matmul/fused) |

### 실행/검증 도구 (tools/execution.py)

| Tool | 설명 |
|------|------|
| `check_gpu_status()` | GPU 가용성 확인 |
| `run_triton_kernel(code, test_input_code)` | 커널 실행 |
| `validate_correctness(kernel_code, reference_code, test_input_code, rtol, atol)` | torch 결과와 비교 검증 |
| `benchmark_kernel(kernel_code, test_input_code, reference_code, warmup, rep)` | 성능 측정 |

---

## 4. 핵심 설계 결정

### 역할 분리

- **LLM이 코드 작성** → MCP 도구는 정보 제공 + 실행/검증만 담당
- MCP 도구가 코드를 생성하지 않음

### TritonRunner (utils/runner.py)

- `exec()`로 코드 문자열을 동적 실행
- GPU 없으면 문법 검사만 수행하고 실행 거부
- CUDA 이벤트로 정확한 시간 측정
- stdout/stderr 캡처

### FastMCP 사용

```python
from fastmcp import FastMCP
mcp = FastMCP(name="triton-kernel-mcp")

@mcp.tool()
def my_tool():
    ...
```

---

## 5. 다음 단계 (GPU 서버에서 할 일)

### Step 1: 의존성 설치

```bash
cd /root/mcp-for-triton-kernel
uv sync
```

### Step 2: Cursor MCP 설정

`.cursor/mcp.json` 생성:

```json
{
  "mcpServers": {
    "triton-kernel": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/root/mcp-for-triton-kernel",
        "python",
        "-m",
        "mcp_for_triton_kernel.server"
      ]
    }
  }
}
```

### Step 3: 테스트

1. Cursor 재시작
2. MCP 도구 연결 확인
3. `vector_add` 예제로 end-to-end 테스트:
    - `get_overview()` 호출
    - `get_torch_op_info("vector_add")` 호출
    - 커널 코드 작성 요청
    - `run_triton_kernel()` 실행
    - `validate_correctness()` 검증

### Step 4: 필요시 확장

- `torch_ops.json`에 더 많은 연산 추가
- 에러 메시지 파싱 개선
- autotune 지원 등

---

## 6. 주요 파일 요약

### server.py

MCP 서버 메인. `FastMCP` 인스턴스 생성하고 도구 등록.

### tools/info.py

- `load_knowledge()`: knowledge/ 폴더에서 md/json 파일 로드
- 4개 정보 도구 구현

### tools/execution.py

- `get_runner()`: 전역 TritonRunner 인스턴스 반환
- 4개 실행/검증 도구 구현
- GPU 없으면 graceful 에러 반환

### utils/runner.py

- `KernelResult`, `ValidationResult`, `BenchmarkResult` 데이터클래스
- `TritonRunner` 클래스:
    - `execute_code()`: 코드 실행
    - `validate_correctness()`: allclose 검증
    - `benchmark()`: 성능 측정

### knowledge/torch_ops.json

12개 연산 정보: vector_add, vector_mul, relu, gelu, softmax, layer_norm, matmul, dropout, attention, cross_entropy, sum, mean

---

## 7. 요구사항

| 패키지 | 버전 |
|--------|------|
| Python | >= 3.10 |
| PyTorch | >= 2.0.0 |
| Triton | >= 2.1.0 |
| fastmcp | >= 2.14.0 |
| CUDA GPU | 실행 시 필수 |

---

## 8. 파일 읽기 권장 순서

1. `README.md` - 전체 개요
2. `server.py` - MCP 서버 구조
3. `tools/info.py` - 정보 도구 구현
4. `tools/execution.py` - 실행 도구 구현
5. `utils/runner.py` - 커널 실행 로직
6. `knowledge/torch_ops.json` - 지원 연산 목록

