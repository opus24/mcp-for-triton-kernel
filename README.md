# mcp-for-triton-kernel

PyTorch 연산을 Triton 커널로 변환하는 작업을 지원하는 MCP (Model Context Protocol) 서버입니다.

## 설치

```bash
# 프로젝트 클론
git clone <repo-url>
cd mcp-for-triton-kernel

# 의존성 설치 (uv 사용)
uv sync

# 또는 pip 사용
pip install -e .
```

## Cursor IDE 설정

`.cursor/mcp.json` 파일을 생성하고 다음 내용을 추가하세요:

```json
{
  "mcpServers": {
    "triton-kernel": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/mcp-for-triton-kernel",
        "python",
        "-m",
        "mcp_for_triton_kernel.server"
      ]
    }
  }
}
```

`/path/to/mcp-for-triton-kernel`를 실제 프로젝트 경로로 변경하세요.

## 제공하는 MCP Tools

### 정보 제공 도구

| Tool | 설명 |
|------|------|
| `get_overview` | Triton 커널 개발 전체 프로세스 안내 |
| `get_triton_syntax` | Triton 문법, tl 함수, 제약사항 레퍼런스 |
| `get_torch_op_info(op_name)` | PyTorch 연산 정보 조회 |
| `get_kernel_template(pattern)` | 커널 템플릿 제공 (elementwise, reduction, matmul, fused) |

### 실행/검증 도구

| Tool | 설명 |
|------|------|
| `check_gpu_status` | GPU 가용성 및 상태 확인 |
| `run_triton_kernel(code, test_input_code)` | Triton 커널 실행 |
| `validate_correctness(kernel_code, reference_code, test_input_code)` | 정확성 검증 |
| `benchmark_kernel(kernel_code, test_input_code, reference_code)` | 성능 측정 |

## 사용 워크플로우

```
1. get_overview()           → 전체 프로세스 파악
2. get_torch_op_info("op")  → 변환할 연산 정보 확인
3. get_triton_syntax()      → Triton 문법 참고
4. (LLM이 커널 코드 작성)
5. run_triton_kernel()      → 실행 테스트
6. validate_correctness()   → 정확성 검증
7. benchmark_kernel()       → 성능 측정
```

## 지원하는 연산

- `vector_add`, `vector_mul` - 요소별 연산
- `relu`, `gelu` - 활성화 함수
- `softmax`, `layer_norm` - 정규화
- `matmul` - 행렬 곱셈
- `dropout` - 드롭아웃
- `attention` - 어텐션
- `cross_entropy` - 손실 함수
- `sum`, `mean` - 축소 연산

## 프로젝트 구조

```
mcp_for_triton_kernel/
├── __init__.py
├── __main__.py           # python -m 실행 지원
├── server.py             # MCP 서버 엔트리포인트
├── tools/
│   ├── info.py           # 정보 제공 도구들
│   └── execution.py      # 실행/검증 도구들
├── knowledge/
│   ├── overview.md       # Triton 개발 가이드
│   ├── torch_ops.json    # Torch ops 정보
│   └── triton_syntax.md  # Triton 문법 정리
└── utils/
    └── runner.py         # 커널 실행 유틸리티
```

## 요구사항

- Python >= 3.10
- PyTorch >= 2.0.0
- Triton >= 2.1.0
- CUDA GPU (실행 시)

## 라이선스

MIT
