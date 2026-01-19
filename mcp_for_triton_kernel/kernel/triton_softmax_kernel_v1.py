import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    N,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Basic softmax kernel with numerical stability (no optimizations)."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load row data
    row_start = row_idx * stride_row
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))

    # Numerical stability: subtract max
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max

    # Compute exp
    x_exp = tl.exp(x_shifted)

    # Compute sum
    x_sum = tl.sum(x_exp, axis=0)

    # Normalize
    output = x_exp / x_sum

    # Store output
    tl.store(output_ptr + row_start + col_offsets, output, mask=mask)


def solve(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Wrapper function to call the softmax kernel.

    Args:
        x: Input tensor
        dim: Dimension along which softmax is computed (default: -1)

    Returns:
        Output tensor with softmax applied along dim
    """
    assert x.device.type == "cuda", "Input tensor must be on CUDA"

    # Handle negative dim
    if dim < 0:
        dim = x.dim() + dim

    # For simplicity, handle 2D case with dim=-1 (last dimension)
    # This can be extended for other cases
    if x.dim() == 2 and dim == 1:
        output = torch.empty_like(x)
        num_rows, num_cols = x.shape

        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        stride_row = x.stride(0)

        # Fixed BLOCK_SIZE (no autotune)
        BLOCK_SIZE = 256
        grid = (num_rows,)

        softmax_kernel[grid](x, output, num_cols, stride_row, BLOCK_SIZE=BLOCK_SIZE)
        return output
    else:
        # Fallback to PyTorch for other cases
        return torch.nn.functional.softmax(x, dim=dim)
