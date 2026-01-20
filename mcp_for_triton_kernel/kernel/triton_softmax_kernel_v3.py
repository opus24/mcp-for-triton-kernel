import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ],
    key=["n_cols"],
)
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax kernel - v3 (Autotune for optimal BLOCK_SIZE)"""
    row_idx = tl.program_id(0)

    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_data = tl.load(input_row_start + col_offsets, mask=mask, other=-float("inf"))

    # Numerically stable softmax
    row_max = tl.max(row_data, axis=0)
    numerator = tl.exp(row_data - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_row_start + col_offsets, softmax_output, mask=mask)


def solve(x: torch.Tensor) -> torch.Tensor:
    """Entry point for softmax operation"""
    x = x.contiguous()
    original_shape = x.shape

    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    output = torch.empty_like(x_2d)

    grid = (n_rows,)

    softmax_kernel[grid](
        x_2d,
        output,
        n_rows,
        n_cols,
        x_2d.stride(0),
        output.stride(0),
    )

    return output.view(original_shape)
