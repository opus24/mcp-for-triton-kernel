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
    """Softmax kernel - v2 (Online Reduction for large rows)

    Uses online algorithm to compute max and sum in chunks,
    enabling processing of rows larger than BLOCK_SIZE.
    """
    row_idx = tl.program_id(0)

    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    # Online reduction: compute max and sum in one pass
    m_i = -float("inf")  # running max
    l_i = 0.0  # running sum of exp

    # First pass: compute online max and sum
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        x = tl.load(input_row_start + col_offsets, mask=mask, other=-float("inf"))

        # Online max update
        m_ij = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, m_ij)

        # Rescale previous sum and add new contribution
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    # Second pass: normalize with final max and sum
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        x = tl.load(input_row_start + col_offsets, mask=mask, other=-float("inf"))
        softmax_output = tl.exp(x - m_i) / l_i
        tl.store(output_row_start + col_offsets, softmax_output, mask=mask)


def solve(x: torch.Tensor) -> torch.Tensor:
    """Entry point for softmax operation"""
    x = x.contiguous()
    original_shape = x.shape

    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    output = torch.empty_like(x_2d)

    # Fixed block size for chunked processing
    BLOCK_SIZE = 1024

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
