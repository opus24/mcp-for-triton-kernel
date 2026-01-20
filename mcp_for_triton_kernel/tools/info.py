"""Information tools for Triton kernel development."""

import json
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from ..knowledge import KNOWLEDGE_DIR, load_knowledge
from ..state import Status, get_state_manager, log_tool_call

# 최적화 기법 카탈로그
OPTIMIZATION_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "Online Reduction",
        "description": "Flash Attention 스타일의 online max/sum 계산. 여러 패스를 한 번에 처리하여 메모리 읽기 최소화. 새로운 max 발견 시 sum rescale.",
        "applies_to": ["softmax", "layernorm", "attention", "cross_entropy", "sum", "mean"],
        "keywords": ["reduction", "softmax", "max", "sum", "mean", "normalize"],
        "memory_patterns": ["row-wise reduction", "reduction"],
    },
    {
        "name": "Autotune",
        "description": "BLOCK_SIZE와 num_warps를 자동으로 튜닝하여 다양한 입력 크기에 최적 성능 달성.",
        "applies_to": ["all"],
        "keywords": ["any"],
        "memory_patterns": ["any"],
    },
    {
        "name": "Welford 알고리즘",
        "description": "Mean과 Variance를 한 번의 패스로 동시에 계산. 메모리 읽기 2배 감소.",
        "applies_to": ["layernorm", "batchnorm", "variance", "std"],
        "keywords": ["mean", "variance", "std", "normalize", "layer", "batch"],
        "memory_patterns": ["row-wise reduction", "reduction"],
    },
    {
        "name": "Tiled Processing",
        "description": "큰 행렬을 작은 타일(BLOCK_M × BLOCK_N × BLOCK_K)로 나눠 처리. 캐시 효율 향상, 큰 행렬도 처리 가능.",
        "applies_to": ["matmul", "attention", "conv"],
        "keywords": ["matmul", "matrix", "gemm", "attention", "conv"],
        "memory_patterns": ["타일링", "shared memory", "2D"],
    },
    {
        "name": "Register Blocking",
        "description": "중간 결과를 레지스터에 유지하여 메모리 접근 최소화. 연산 강도 증가.",
        "applies_to": ["matmul", "attention", "conv"],
        "keywords": ["matmul", "matrix", "gemm", "attention"],
        "memory_patterns": ["타일링", "shared memory"],
    },
    {
        "name": "Coalesced Memory Access",
        "description": "연속된 메모리 접근 패턴 유지, stride 최소화. 메모리 bandwidth 활용 극대화.",
        "applies_to": ["all"],
        "keywords": ["element", "vector", "add", "mul", "div"],
        "memory_patterns": ["element-wise", "완벽하게 병렬화"],
    },
    {
        "name": "Kernel Specialization",
        "description": "tl.constexpr로 컴파일 타임 상수화, 조건부 컴파일. 불필요한 분기 제거, 최적화된 코드 생성.",
        "applies_to": ["dropout", "gelu", "cross_entropy"],
        "keywords": ["random", "activation", "special", "conditional"],
        "memory_patterns": ["element-wise with random", "element-wise"],
    },
]


def _analyze_operation(
    op_name: str, description: str = "", memory_pattern: str = ""
) -> List[Dict[str, Any]]:
    """
    연산 특성을 분석하여 적합한 최적화 기법 2개를 선택합니다.

    Args:
        op_name: 연산 이름
        description: 연산 설명
        memory_pattern: 메모리 접근 패턴

    Returns:
        추천된 최적화 기법 리스트 (2개)
    """
    scores: Dict[str, float] = {}
    op_name_lower = op_name.lower()
    description_lower = description.lower()
    memory_pattern_lower = memory_pattern.lower()

    for tech in OPTIMIZATION_CATALOG:
        score = 0.0
        tech_name = tech["name"]

        # 1. applies_to 매칭 (높은 가중치)
        if "all" in tech["applies_to"]:
            score += 1.0
        for applies in tech["applies_to"]:
            if applies.lower() in op_name_lower:
                score += 5.0

        # 2. 키워드 매칭
        for keyword in tech["keywords"]:
            if keyword == "any":
                score += 0.5
            elif keyword in op_name_lower:
                score += 3.0
            elif keyword in description_lower:
                score += 2.0

        # 3. 메모리 패턴 매칭 (높은 가중치)
        for pattern in tech["memory_patterns"]:
            if pattern == "any":
                score += 0.5
            elif pattern.lower() in memory_pattern_lower:
                score += 4.0

        scores[tech_name] = score

    # 점수 기준 정렬
    sorted_techs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 상위 2개 선택 (Autotune이 항상 포함되도록 보장)
    selected = []
    autotune_tech = None

    for tech_name, score in sorted_techs:
        if tech_name == "Autotune":
            autotune_tech = next(t for t in OPTIMIZATION_CATALOG if t["name"] == tech_name)
        elif len(selected) < 1 and score > 0:
            tech = next(t for t in OPTIMIZATION_CATALOG if t["name"] == tech_name)
            selected.append(
                {
                    "name": tech["name"],
                    "description": tech["description"],
                    "applies_to": tech["applies_to"],
                }
            )

    # 첫 번째가 없으면 Coalesced Memory Access 추가
    if len(selected) == 0:
        coalesced = next(t for t in OPTIMIZATION_CATALOG if t["name"] == "Coalesced Memory Access")
        selected.append(
            {
                "name": coalesced["name"],
                "description": coalesced["description"],
                "applies_to": coalesced["applies_to"],
            }
        )

    # Autotune 항상 두 번째로 추가
    if autotune_tech:
        selected.append(
            {
                "name": autotune_tech["name"],
                "description": autotune_tech["description"],
                "applies_to": autotune_tech["applies_to"],
            }
        )

    return selected[:2]


def register_info_tools(mcp: FastMCP) -> None:
    """Register information-providing tools to the MCP server."""

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.START])
    def get_overview() -> str:
        """
        Triton 커널 개발의 전체 프로세스와 기본 구조를 설명합니다.

        이 도구는 'start' 상태에서만 사용할 수 있습니다.
        커널 개발을 시작하기 전에 이 도구를 호출하여 전체적인 흐름을 파악하세요.

        Returns:
            Triton 커널 개발 가이드 문서
        """
        state = get_state_manager()
        state.mark_info_collected("get_overview")

        content = load_knowledge("overview.md")

        status_hint = ""
        if state.can_transition_to_write():
            status_hint = "\n\n✅ 모든 정보 수집 완료! 상태가 'write'로 전환되었습니다."
        else:
            missing = [t for t, done in state.info_collected.items() if not done]
            status_hint = f"\n\n📋 아직 수집이 필요한 정보: {', '.join(missing)}"

        return content + status_hint

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.START])
    def get_triton_syntax() -> str:
        """
        Triton 문법, tl 함수들, 제약사항에 대한 레퍼런스를 제공합니다.

        이 도구는 'start' 상태에서만 사용할 수 있습니다.
        커널 코드를 작성할 때 참고하세요.

        Returns:
            Triton 문법 레퍼런스 문서
        """
        state = get_state_manager()
        state.mark_info_collected("get_triton_syntax")

        content = load_knowledge("triton_syntax.md")

        status_hint = ""
        if state.can_transition_to_write():
            status_hint = "\n\n✅ 모든 정보 수집 완료! 상태가 'write'로 전환되었습니다."
        else:
            missing = [t for t, done in state.info_collected.items() if not done]
            status_hint = f"\n\n📋 아직 수집이 필요한 정보: {', '.join(missing)}"

        return content + status_hint

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.START])
    def get_torch_op_info(op_name: Optional[str] = None) -> str:
        """
        PyTorch 연산에 대한 정보를 제공합니다.

        이 도구는 'start' 상태에서만 사용할 수 있습니다.
        특정 연산명을 지정하면 해당 연산의 상세 정보를,
        지정하지 않으면 지원하는 모든 연산 목록을 반환합니다.

        Args:
            op_name: 조회할 연산 이름 (예: "softmax", "matmul", "relu")
                    None이면 전체 목록 반환

        Returns:
            연산 정보 (시그니처, 설명, Triton 구현 팁 등)
        """
        state = get_state_manager()
        state.mark_info_collected("get_torch_op_info")

        torch_ops_path = KNOWLEDGE_DIR / "torch_ops.json"

        if not torch_ops_path.exists():
            return "Error: torch_ops.json not found"

        with open(torch_ops_path, "r", encoding="utf-8") as f:
            ops_data = json.load(f)

        status_hint = ""
        if state.can_transition_to_write():
            status_hint = "\n\n✅ 모든 정보 수집 완료! 상태가 'write'로 전환되었습니다."
        else:
            missing = [t for t, done in state.info_collected.items() if not done]
            status_hint = f"\n\n📋 아직 수집이 필요한 정보: {', '.join(missing)}"

        if op_name is None:
            # 전체 목록 반환
            ops_list = list(ops_data.keys())
            return f"""사용 가능한 연산 목록 ({len(ops_list)}개):

{chr(10).join(f"- {op}" for op in ops_list)}

특정 연산의 상세 정보를 보려면 op_name 인자를 지정하세요.
예: get_torch_op_info("softmax")
{status_hint}"""

        # 정규화된 이름으로 검색
        normalized_name = op_name.lower().strip()

        if normalized_name not in ops_data:
            # 부분 매칭 시도
            matches = [op for op in ops_data.keys() if normalized_name in op.lower()]
            if matches:
                return f"""'{op_name}' 연산을 찾을 수 없습니다.

유사한 연산:
{chr(10).join(f"- {m}" for m in matches)}
{status_hint}"""
            return f"'{op_name}' 연산을 찾을 수 없습니다. get_torch_op_info()로 전체 목록을 확인하세요.{status_hint}"

        op_info = ops_data[normalized_name]

        # 최적화 기법 정보 포맷팅
        optimization_section = ""
        if "optimization_techniques" in op_info and op_info["optimization_techniques"]:
            techniques = op_info["optimization_techniques"]
            optimization_section = "\n## 🚀 추천 최적화 기법 (4가지 버전 생성용)\n\n"
            optimization_section += (
                "다음 2가지 최적화 기법을 조합하여 v1~v4 커널을 만들 수 있습니다:\n\n"
            )

            for i, tech in enumerate(techniques[:2], 1):  # 최대 2개만 표시
                optimization_section += f"### 기법 {i}: {tech['name']}\n"
                optimization_section += f"{tech['description']}\n\n"

            optimization_section += "**4가지 버전 구성:**\n"
            optimization_section += "- **v1**: 기본 구현 (최적화 없음)\n"
            optimization_section += f"- **v2**: {techniques[0]['name']}만 적용\n"
            if len(techniques) > 1:
                optimization_section += f"- **v3**: {techniques[1]['name']}만 적용\n"
                optimization_section += (
                    f"- **v4**: {techniques[0]['name']} + {techniques[1]['name']} 모두 적용\n"
                )
            else:
                optimization_section += f"- **v2~v4**: {techniques[0]['name']}의 다양한 변형\n"

        return f"""# {normalized_name}

## PyTorch 동등 표현
{op_info.get('torch_equivalent', 'N/A')}

## 시그니처
```python
{op_info.get('signature', 'N/A')}
```

## 설명
{op_info.get('description', 'N/A')}

## 입력 Shape
{op_info.get('input_shapes', 'N/A')}

## 출력 Shape
{op_info.get('output_shape', 'N/A')}

## 복잡도
{op_info.get('complexity', 'N/A')}

## 메모리 패턴
{op_info.get('memory_pattern', 'N/A')}

## Triton 구현 팁
{op_info.get('triton_tips', 'N/A')}
{optimization_section}{status_hint}"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.START])
    def get_kernel_template(pattern: str = "elementwise") -> str:
        """
        일반적인 Triton 커널 템플릿을 제공합니다.

        이 도구는 'start' 상태에서만 사용할 수 있습니다.

        Args:
            pattern: 커널 패턴 종류
                - "elementwise": 요소별 연산 (add, mul, relu 등)
                - "reduction": 축소 연산 (sum, mean, max 등)
                - "matmul": 행렬 곱셈
                - "fused": 융합 커널 (예: softmax)

        Returns:
            해당 패턴의 커널 템플릿 코드
        """
        templates = {
            "elementwise": '''import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise operation kernel template."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)

    # TODO: 여기에 연산 구현
    # 예: y = x * 2, y = tl.where(x > 0, x, 0), etc.
    y = x

    # Store output
    tl.store(output_ptr + offsets, y, mask=mask)


def solve(input: torch.Tensor) -> torch.Tensor:
    """Wrapper function to call the kernel."""
    output = torch.empty_like(input)
    N = input.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    elementwise_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)

    return output
''',
            "reduction": '''import torch
import triton
import triton.language as tl


@triton.jit
def reduction_kernel(
    input_ptr,
    output_ptr,
    M,  # number of rows
    N,  # number of columns (reduction dimension)
    stride_m,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-wise reduction kernel template."""
    row_idx = tl.program_id(0)

    # Initialize accumulator
    acc = 0.0

    # Iterate over columns in blocks
    for start in range(0, N, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # Load data
        ptrs = input_ptr + row_idx * stride_m + col_offsets
        x = tl.load(ptrs, mask=mask, other=0.0)

        # TODO: Accumulate (change operation as needed)
        acc += tl.sum(x, axis=0)

    # Store result
    tl.store(output_ptr + row_idx, acc)


def solve(input: torch.Tensor) -> torch.Tensor:
    """Wrapper function to call the kernel."""
    M, N = input.shape
    output = torch.empty(M, device=input.device, dtype=input.dtype)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 1024)

    grid = (M,)
    reduction_kernel[grid](
        input, output, M, N, input.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
''',
            "matmul": '''import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Matrix multiplication kernel: C = A @ B."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block starting positions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first block of A and B
    A_block_ptr = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_block_ptr = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        A_block = tl.load(A_block_ptr, mask=offs_m[:, None] < M and k_mask[None, :], other=0.0)
        B_block = tl.load(B_block_ptr, mask=k_mask[:, None] and offs_n[None, :] < N, other=0.0)

        acc += tl.dot(A_block, B_block)

        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk

    # Store result
    C_block_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_block_ptr, acc.to(C_ptr.dtype.element_ty), mask=mask)


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper function to call the kernel."""
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return C
''',
            "fused": '''import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    M,  # number of rows
    N,  # number of columns
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused softmax kernel (numerically stable)."""
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * stride
    out_start = output_ptr + row_idx * stride

    # Load row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    x = tl.load(row_start + col_offsets, mask=mask, other=float("-inf"))

    # Compute softmax (numerically stable)
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp

    # Store
    tl.store(out_start + col_offsets, softmax, mask=mask)


def solve(input: torch.Tensor) -> torch.Tensor:
    """Wrapper function to call the kernel."""
    M, N = input.shape
    output = torch.empty_like(input)

    # BLOCK_SIZE must be >= N for this simple version
    BLOCK_SIZE = triton.next_power_of_2(N)
    assert BLOCK_SIZE <= 2048, "Row too large for single-block softmax"

    grid = (M,)
    fused_softmax_kernel[grid](
        input, output, M, N, input.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
''',
        }

        if pattern not in templates:
            available = ", ".join(templates.keys())
            return f"Unknown pattern: {pattern}\nAvailable patterns: {available}"

        return templates[pattern]

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.START])
    def analyze_and_save_optimization(
        op_name: str,
        torch_equivalent: str,
        signature: str,
        description: str,
        input_shapes: str,
        output_shape: str,
        complexity: str,
        memory_pattern: str,
        triton_tips: str,
    ) -> str:
        """
        연산을 분석하여 적합한 최적화 기법 2개를 선택하고 torch_ops.json에 저장합니다.

        이 도구는 'start' 상태에서만 사용할 수 있습니다.
        새로운 연산을 등록하거나 기존 연산의 최적화 기법을 업데이트할 때 사용합니다.

        Args:
            op_name: 연산 이름 (예: "softmax", "matmul", "relu")
            torch_equivalent: PyTorch 동등 표현 (예: "torch.nn.functional.softmax(x, dim=-1)")
            signature: 함수 시그니처 (예: "softmax(x: Tensor, dim: int = -1) -> Tensor")
            description: 연산 설명 (예: "exp(x - max(x)) / sum(exp(x - max(x)))")
            input_shapes: 입력 shape 정보 (예: "보통 (batch, seq_len, hidden)")
            output_shape: 출력 shape 정보 (예: "입력과 동일")
            complexity: 복잡도 (예: "O(N) per row")
            memory_pattern: 메모리 접근 패턴 (예: "row-wise reduction 필요")
            triton_tips: Triton 구현 팁

        Returns:
            분석 결과 및 저장 결과
        """
        state = get_state_manager()

        # 연산 분석하여 최적화 기법 선택
        optimization_techniques = _analyze_operation(
            op_name=op_name,
            description=description,
            memory_pattern=memory_pattern,
        )

        # 새 연산 데이터 구성
        op_data = {
            "torch_equivalent": torch_equivalent,
            "signature": signature,
            "description": description,
            "input_shapes": input_shapes,
            "output_shape": output_shape,
            "complexity": complexity,
            "memory_pattern": memory_pattern,
            "triton_tips": triton_tips,
            "optimization_techniques": optimization_techniques,
        }

        # torch_ops.json 읽기
        torch_ops_path = KNOWLEDGE_DIR / "torch_ops.json"

        if torch_ops_path.exists():
            with open(torch_ops_path, "r", encoding="utf-8") as f:
                ops_data = json.load(f)
        else:
            ops_data = {}

        # 연산 추가/업데이트
        normalized_name = op_name.lower().strip()
        is_update = normalized_name in ops_data
        ops_data[normalized_name] = op_data

        # torch_ops.json 저장
        with open(torch_ops_path, "w", encoding="utf-8") as f:
            json.dump(ops_data, f, ensure_ascii=False, indent=2)

        # 상태 힌트
        status_hint = ""
        if state.can_transition_to_write():
            status_hint = "\n\n✅ 모든 정보 수집 완료! 상태가 'write'로 전환되었습니다."
        else:
            missing = [t for t, done in state.info_collected.items() if not done]
            status_hint = f"\n\n📋 아직 수집이 필요한 정보: {', '.join(missing)}"

        # 결과 포맷팅
        action = "업데이트" if is_update else "추가"
        techniques_str = "\n".join(
            [
                f"  {i+1}. **{tech['name']}**: {tech['description']}"
                for i, tech in enumerate(optimization_techniques)
            ]
        )

        return f"""# ✅ 연산 분석 완료: {normalized_name}

## 📊 분석 결과

### 추천 최적화 기법 (2개)
{techniques_str}

### 4가지 버전 구성 가이드
- **v1**: 기본 구현 (최적화 없음)
- **v2**: {optimization_techniques[0]['name']}만 적용
- **v3**: {optimization_techniques[1]['name'] if len(optimization_techniques) > 1 else optimization_techniques[0]['name']}만 적용
- **v4**: {optimization_techniques[0]['name']} + {optimization_techniques[1]['name'] if len(optimization_techniques) > 1 else '추가 최적화'} 모두 적용

## 💾 저장 결과

- **파일**: `{torch_ops_path}`
- **작업**: {action}
- **연산명**: `{normalized_name}`
{status_hint}"""

    @mcp.tool()
    @log_tool_call(allowed_statuses=[Status.START])
    def get_optimization_catalog() -> str:
        """
        사용 가능한 모든 최적화 기법 카탈로그를 반환합니다.

        이 도구는 'start' 상태에서만 사용할 수 있습니다.
        어떤 최적화 기법들이 있는지 확인할 때 사용합니다.

        Returns:
            최적화 기법 목록 및 설명
        """
        state = get_state_manager()

        catalog_str = ""
        for i, tech in enumerate(OPTIMIZATION_CATALOG, 1):
            applies_to = ", ".join(tech["applies_to"])
            catalog_str += f"""
### {i}. {tech["name"]}

**설명**: {tech["description"]}

**적용 대상**: {applies_to}

---
"""

        status_hint = ""
        if state.can_transition_to_write():
            status_hint = "\n\n✅ 모든 정보 수집 완료! 상태가 'write'로 전환되었습니다."
        else:
            missing = [t for t, done in state.info_collected.items() if not done]
            status_hint = f"\n\n📋 아직 수집이 필요한 정보: {', '.join(missing)}"

        return f"""# 🚀 Triton 커널 최적화 기법 카탈로그

총 {len(OPTIMIZATION_CATALOG)}개의 최적화 기법이 등록되어 있습니다.
{catalog_str}

## 사용법

`analyze_and_save_optimization` 도구를 사용하면 연산 특성을 분석하여
자동으로 적합한 최적화 기법 2개를 선택하고 `torch_ops.json`에 저장합니다.
{status_hint}"""
