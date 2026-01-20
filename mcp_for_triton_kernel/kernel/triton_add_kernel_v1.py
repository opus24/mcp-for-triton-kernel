import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    A_ptr,
    B_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Basic vector add kernel - v1 (no optimization)"""
    # Get program ID
    pid = tl.program_id(0)

    # Calculate offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create mask for boundary check
    mask = offsets < N

    # Load data
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)

    # Perform addition
    result = a + b

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point for vector add operation"""
    # Ensure inputs are contiguous and on CUDA
    A = A.contiguous()
    B = B.contiguous()

    # Get size
    N = A.numel()

    # Allocate output
    output = torch.empty_like(A)

    # Define block size
    BLOCK_SIZE = 1024

    # Calculate grid size
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # Launch kernel
    add_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)

    return output
