import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise addition kernel: output = a + b."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load inputs
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Perform addition
    result = a + b
    
    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper function to call the kernel."""
    # Ensure inputs have the same shape
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert a.device == b.device, "Inputs must be on the same device"
    
    output = torch.empty_like(a)
    N = a.numel()
    BLOCK_SIZE = 256
    
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, output, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


