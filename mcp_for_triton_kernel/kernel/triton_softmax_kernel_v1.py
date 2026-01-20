import torch
import triton
import triton.language as tl


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
    """Softmax kernel - v1 (basic 3-pass implementation)"""
    # Get row index
    row_idx = tl.program_id(0)

    # Calculate row start pointers
    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row data
    row_data = tl.load(input_row_start + col_offsets, mask=mask, other=-float("inf"))

    # Pass 1: Find max for numerical stability
    row_max = tl.max(row_data, axis=0)

    # Pass 2: Compute exp(x - max)
    numerator = tl.exp(row_data - row_max)

    # Pass 3: Compute sum and normalize
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store result
    tl.store(output_row_start + col_offsets, softmax_output, mask=mask)


def solve(x: torch.Tensor) -> torch.Tensor:
    """Entry point for softmax operation"""
    x = x.contiguous()
    original_shape = x.shape

    # Reshape to 2D
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    output = torch.empty_like(x_2d)

    # BLOCK_SIZE must be >= n_cols (power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # One program per row
    grid = (n_rows,)

    softmax_kernel[grid](
        x_2d,
        output,
        n_rows,
        n_cols,
        x_2d.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(original_shape)
