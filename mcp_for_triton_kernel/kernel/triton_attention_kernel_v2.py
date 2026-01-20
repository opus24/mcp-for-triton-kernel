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
    """Attention kernel - v2 (Online Reduction with block key processing)

    Processes keys in blocks for better memory access patterns.
    """
    pid = tl.program_id(0)
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    q_pos = rem % S

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load query vector
    q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + q_pos * stride_qs + d_offs * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Online softmax state
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Process keys in blocks
    s_offs = tl.arange(0, BLOCK_S)

    for k_start in range(0, S, BLOCK_S):
        # Compute scores for this block of keys
        scores = tl.zeros([BLOCK_S], dtype=tl.float32) - float("inf")

        for i in range(BLOCK_S):
            k_pos = k_start + i
            if k_pos < S:
                k_ptrs = (
                    K_ptr + b * stride_kb + h * stride_kh + k_pos * stride_ks + d_offs * stride_kd
                )
                k = tl.load(k_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                score = tl.sum(q * k, axis=0) * scale
                scores = tl.where(s_offs == i, score, scores)

        # Online softmax update for this block
        block_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)

        acc = acc * alpha
        l_i = l_i * alpha

        for i in range(BLOCK_S):
            k_pos = k_start + i
            if k_pos < S:
                score = tl.where(s_offs == i, scores, 0.0)
                score = tl.sum(score, axis=0)
                p = tl.exp(score - m_new)
                l_i += p

                v_ptrs = (
                    V_ptr + b * stride_vb + h * stride_vh + k_pos * stride_vs + d_offs * stride_vd
                )
                v = tl.load(v_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                acc += p * v

        m_i = m_new

    # Normalize
    out = acc / l_i

    # Store output
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

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_S = min(16, S)  # Process 16 keys at a time

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
