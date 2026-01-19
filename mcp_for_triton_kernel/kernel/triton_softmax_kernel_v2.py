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
    """Softmax kernel with Online Reduction optimization (no autotune)."""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_row

    # Calculate number of chunks (ceiling division)
    num_chunks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Pass 1: Online max and sum computation
    # Initialize running max and sum
    running_max = float("-inf")
    running_sum = 0.0

    # Process data in chunks
    for chunk_idx in range(num_chunks):
        col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # Load chunk
        chunk = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))

        # Compute chunk max (scalar reduction)
        chunk_max = tl.max(chunk, axis=0)

        # Update running max and rescale sum if needed
        # Online softmax: when new max is found, rescale existing sum
        old_max = running_max
        running_max = max(running_max, chunk_max)

        # Rescale sum when new max is found (if old_max == running_max, exp(0)=1, so no change)
        rescale_factor = tl.exp(old_max - running_max)
        running_sum = running_sum * rescale_factor

        # Compute exp of shifted chunk and add to sum
        chunk_shifted = chunk - running_max
        chunk_exp = tl.exp(chunk_shifted)
        chunk_sum = tl.sum(chunk_exp, axis=0)
        running_sum = running_sum + chunk_sum

    # Pass 2: Normalize and store
    for chunk_idx in range(num_chunks):
        col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # Load chunk again
        chunk = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))

        # Compute exp of shifted chunk
        chunk_shifted = chunk - running_max
        chunk_exp = tl.exp(chunk_shifted)

        # Normalize
        output = chunk_exp / running_sum

        # Store output
        tl.store(output_ptr + row_start + col_offsets, output, mask=mask)


def solve(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Wrapper function to call the Online Softmax kernel.

    This implementation uses Online Reduction (Flash Attention style) to optimize
    memory access by processing data in chunks and computing max/sum incrementally.

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
