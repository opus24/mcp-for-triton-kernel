"""Information tools for Triton kernel development."""

import json
from typing import Optional

from fastmcp import FastMCP

from ..knowledge import load_knowledge, KNOWLEDGE_DIR


def register_info_tools(mcp: FastMCP) -> None:
    """Register information-providing tools to the MCP server."""

    @mcp.tool()
    def get_overview() -> str:
        """
        Triton 커널 개발의 전체 프로세스와 기본 구조를 설명합니다.
        
        커널 개발을 시작하기 전에 이 도구를 호출하여 전체적인 흐름을 파악하세요.
        
        Returns:
            Triton 커널 개발 가이드 문서
        """
        return load_knowledge("overview.md")

    @mcp.tool()
    def get_triton_syntax() -> str:
        """
        Triton 문법, tl 함수들, 제약사항에 대한 레퍼런스를 제공합니다.
        
        커널 코드를 작성할 때 참고하세요.
        
        Returns:
            Triton 문법 레퍼런스 문서
        """
        return load_knowledge("triton_syntax.md")

    @mcp.tool()
    def get_torch_op_info(op_name: Optional[str] = None) -> str:
        """
        PyTorch 연산에 대한 정보를 제공합니다.
        
        특정 연산명을 지정하면 해당 연산의 상세 정보를,
        지정하지 않으면 지원하는 모든 연산 목록을 반환합니다.
        
        Args:
            op_name: 조회할 연산 이름 (예: "softmax", "matmul", "relu")
                    None이면 전체 목록 반환
        
        Returns:
            연산 정보 (시그니처, 설명, Triton 구현 팁 등)
        """
        torch_ops_path = KNOWLEDGE_DIR / "torch_ops.json"
        
        if not torch_ops_path.exists():
            return "Error: torch_ops.json not found"
        
        with open(torch_ops_path, "r", encoding="utf-8") as f:
            ops_data = json.load(f)
        
        if op_name is None:
            # 전체 목록 반환
            ops_list = list(ops_data.keys())
            return f"""사용 가능한 연산 목록 ({len(ops_list)}개):

{chr(10).join(f"- {op}" for op in ops_list)}

특정 연산의 상세 정보를 보려면 op_name 인자를 지정하세요.
예: get_torch_op_info("softmax")
"""
        
        # 정규화된 이름으로 검색
        normalized_name = op_name.lower().strip()
        
        if normalized_name not in ops_data:
            # 부분 매칭 시도
            matches = [op for op in ops_data.keys() if normalized_name in op.lower()]
            if matches:
                return f"""'{op_name}' 연산을 찾을 수 없습니다.

유사한 연산:
{chr(10).join(f"- {m}" for m in matches)}
"""
            return f"'{op_name}' 연산을 찾을 수 없습니다. get_torch_op_info()로 전체 목록을 확인하세요."
        
        op_info = ops_data[normalized_name]
        
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
"""

    @mcp.tool()
    def get_kernel_template(pattern: str = "elementwise") -> str:
        """
        일반적인 Triton 커널 템플릿을 제공합니다.
        
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

