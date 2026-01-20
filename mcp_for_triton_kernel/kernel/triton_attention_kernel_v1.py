import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    B,
    H,
    S,
    D,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Attention kernel - v1 (basic implementation)

    Computes: softmax(Q @ K.T * scale) @ V
    Each program handles one (batch, head, query_pos) combination.
    """
    # Get batch, head, and query position
    pid = tl.program_id(0)
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    q_pos = rem % S

    # Offsets for D dimension
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load query vector: Q[b, h, q_pos, :]
    q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + q_pos * stride_qs + d_offs * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0)

    # Compute attention scores and accumulate output
    m_i = -float("inf")  # max for numerical stability
    l_i = 0.0  # sum of exp
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Iterate over all key positions
    for k_pos in range(S):
        # Load key vector: K[b, h, k_pos, :]
        k_ptrs = K_ptr + b * stride_kb + h * stride_kh + k_pos * stride_ks + d_offs * stride_kd
        k = tl.load(k_ptrs, mask=d_mask, other=0.0)

        # Compute attention score: Q @ K.T * scale
        score = tl.sum(q * k, axis=0) * scale

        # Online softmax update
        m_new = tl.maximum(m_i, score)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)

        # Rescale accumulator and add new contribution
        acc = acc * alpha
        l_i = l_i * alpha + p

        # Load value vector: V[b, h, k_pos, :]
        v_ptrs = V_ptr + b * stride_vb + h * stride_vh + k_pos * stride_vs + d_offs * stride_vd
        v = tl.load(v_ptrs, mask=d_mask, other=0.0)

        acc += p * v.to(tl.float32)
        m_i = m_new

    # Normalize by sum of exp
    out = acc / l_i

    # Store output: O[b, h, q_pos, :]
    o_ptrs = O_ptr + b * stride_ob + h * stride_oh + q_pos * stride_os + d_offs * stride_od
    tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=d_mask)


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float = None) -> torch.Tensor:
    """Entry point for attention operation"""
    B, H, S, D = Q.shape

    if scale is None:
        scale = 1.0 / (D**0.5)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    output = torch.empty_like(Q)

    # Block size for D dimension (must be power of 2)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_S = 1  # Process one key at a time in v1

    # One program per (batch, head, query_pos)
    grid = (B * H * S,)

    attention_kernel[grid](
        Q,
        K,
        V,
        output,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        B,
        H,
        S,
        D,
        scale,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    return output
