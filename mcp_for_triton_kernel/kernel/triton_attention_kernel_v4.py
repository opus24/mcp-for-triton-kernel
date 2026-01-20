import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8),
    ],
    key=["S", "D"],
)
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Attention kernel - v4 (Online Reduction + Tiled + Autotune)

    Flash Attention style with autotuned tile sizes.
    """
    pid = tl.program_id(0)
    num_q_tiles = tl.cdiv(S, BLOCK_M)

    bh_idx = pid // num_q_tiles
    q_tile_idx = pid % num_q_tiles
    b = bh_idx // H
    h = bh_idx % H

    q_offs = q_tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offs < S
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load query tile
    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + h * stride_qh
        + q_offs[:, None] * stride_qs
        + d_offs[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    # Online softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Process key-value tiles
    for k_start in range(0, S, BLOCK_N):
        k_offs = k_start + tl.arange(0, BLOCK_N)
        k_mask = k_offs < S

        # Load K tile
        k_ptrs = (
            K_ptr
            + b * stride_kb
            + h * stride_kh
            + k_offs[:, None] * stride_ks
            + d_offs[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Compute scores with tl.dot for efficiency
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(q_mask[:, None] & k_mask[None, :], scores, -float("inf"))

        # Online softmax
        row_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, row_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load V tile and accumulate
        v_ptrs = (
            V_ptr
            + b * stride_vb
            + h * stride_vh
            + k_offs[:, None] * stride_vs
            + d_offs[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Normalize and store
    out = acc / l_i[:, None]
    o_ptrs = (
        O_ptr
        + b * stride_ob
        + h * stride_oh
        + q_offs[:, None] * stride_os
        + d_offs[None, :] * stride_od
    )
    tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=q_mask[:, None] & d_mask[None, :])


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

    # Grid size depends on autotuned BLOCK_M
    def grid(meta):
        return (B * H * triton.cdiv(S, meta["BLOCK_M"]),)

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
        BLOCK_D=BLOCK_D,
    )

    return output
