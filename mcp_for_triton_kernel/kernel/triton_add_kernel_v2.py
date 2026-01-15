import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["N"],
)
@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise addition kernel with autotune."""
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
    """Wrapper function to call the add kernel with autotune.

    Args:
        a: First input tensor
        b: Second input tensor (must have same shape as a)

    Returns:
        Output tensor containing a + b
    """
    # Ensure tensors are on CUDA and have same shape
    assert a.device.type == "cuda", "Input tensors must be on CUDA"
    assert b.device.type == "cuda", "Input tensors must be on CUDA"
    assert a.shape == b.shape, "Input tensors must have the same shape"

    output = torch.empty_like(a)
    N = a.numel()

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, output, N)

    return output
