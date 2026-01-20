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
    """Vector add kernel - v2 (Coalesced Memory Access)"""
    # Get program ID
    pid = tl.program_id(0)

    # Calculate offsets with coalesced access pattern
    # Each thread accesses consecutive memory locations
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for boundary check
    mask = offsets < N

    # Load data with coalesced access
    # Using eviction policy for better cache behavior
    a = tl.load(A_ptr + offsets, mask=mask, eviction_policy="evict_last")
    b = tl.load(B_ptr + offsets, mask=mask, eviction_policy="evict_last")

    # Perform addition
    result = a + b

    # Store result with coalesced access
    tl.store(output_ptr + offsets, result, mask=mask, eviction_policy="evict_first")


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point for vector add operation"""
    # Ensure inputs are contiguous for coalesced access
    A = A.contiguous()
    B = B.contiguous()

    # Get size
    N = A.numel()

    # Allocate output (contiguous by default)
    output = torch.empty_like(A)

    # Optimized block size for coalesced access
    BLOCK_SIZE = 1024

    # Calculate grid size
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # Launch kernel
    add_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)

    return output
