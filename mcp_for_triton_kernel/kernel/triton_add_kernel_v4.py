import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def add_kernel(
    A_ptr,
    B_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Vector add kernel - v4 (Coalesced Memory Access + Autotune)"""
    # Get program ID
    pid = tl.program_id(0)

    # Calculate offsets with coalesced access pattern
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for boundary check
    mask = offsets < N

    # Load data with coalesced access and optimized eviction policy
    a = tl.load(A_ptr + offsets, mask=mask, eviction_policy="evict_last")
    b = tl.load(B_ptr + offsets, mask=mask, eviction_policy="evict_last")

    # Perform addition
    result = a + b

    # Store result with optimized eviction policy
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

    # Calculate grid size (autotune will choose BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # Launch kernel (autotune will select best config)
    add_kernel[grid](A, B, output, N)

    return output
