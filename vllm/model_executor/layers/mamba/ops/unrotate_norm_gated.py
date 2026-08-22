"""Gated group-RMSNorm with the R_v un-rotation fused into its prologue.

vLLM's Mixer2RMSNormGated falls back to torch ops whenever n_groups != 1, which is every
hybrid we serve, so there is no vendor kernel to extend -- this is that kernel, plus
the un-rotation. Semantics match forward_native exactly:

    x := unrotate(x)                    per head, R = diag(signs) @ H_d / sqrt(d)
    x := x * silu(gate)                 gate BEFORE the RMS
    x := x * rsqrt(mean(x^2) + eps)     reduced over the FULL group (many heads)
    y := weight * x

One program per (token, group). The group tile holds several heads, and the rotation is
block-diagonal across them, so the butterfly runs inside the tile with no cross-program
traffic and the RMS reduction sees the whole group.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwht_stage(x, NH: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
    """One in-register FWHT butterfly stage at distance H, per head row."""
    if H < D:
        v = tl.reshape(x, (NH, D // (2 * H), 2, H))
        sg = tl.where(tl.arange(0, 2)[None, None, :, None] == 0, 1.0, -1.0)
        x = tl.reshape(v * sg + tl.flip(v, 2), (NH * D,))
    return x


@triton.jit
def _unrotate_norm_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    signs_ptr,
    o_ptr,
    stride_x: tl.int64,
    stride_g: tl.int64,
    stride_o: tl.int64,
    n_groups: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ROTATE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    grp = tl.program_id(1)
    off = grp * group_size + tl.arange(0, group_size)

    x = tl.load(x_ptr + row * stride_x + off).to(tl.float32)
    if ROTATE:
        NH: tl.constexpr = group_size // head_dim
        # row form of @ R with R = diag(signs) @ H/sqrt(d): out = H (signs*y) / sqrt(d),
        # so the signs go on FIRST and the Hadamard second
        sg = tl.load(signs_ptr + (off % head_dim)).to(tl.float32)
        x = x * sg * (1.0 / tl.sqrt(float(head_dim)))
        x = _fwht_stage(x, NH, head_dim, 1)
        x = _fwht_stage(x, NH, head_dim, 2)
        x = _fwht_stage(x, NH, head_dim, 4)
        x = _fwht_stage(x, NH, head_dim, 8)
        x = _fwht_stage(x, NH, head_dim, 16)
        x = _fwht_stage(x, NH, head_dim, 32)
        x = _fwht_stage(x, NH, head_dim, 64)

    g = tl.load(g_ptr + row * stride_g + off).to(tl.float32)
    x = x * (g / (1.0 + tl.exp(-g)))  # silu gate, before the RMS
    x = x * tl.rsqrt(tl.sum(x * x, 0) / group_size + eps)
    if HAS_WEIGHT:
        x = x * tl.load(w_ptr + off).to(tl.float32)
    tl.store(o_ptr + row * stride_o + off, x.to(o_ptr.dtype.element_ty))


def unrotate_rmsnorm_gated(x, gate, weight, signs, head_dim, group_size, eps):
    """Fused un-rotate + gated group-RMSNorm. `signs=None` skips the rotation."""
    # the gate is a SLICE of projected_states in the mixer, so its row stride is NOT the
    # hidden size -- carry x, gate and out strides separately or every row past the first
    # reads the wrong gate
    assert x.shape == gate.shape
    assert x.stride(-1) == 1 and gate.stride(-1) == 1
    hidden = x.shape[-1]
    rows = x.numel() // hidden
    assert hidden % group_size == 0, (hidden, group_size)
    assert signs is None or group_size % head_dim == 0, (group_size, head_dim)

    out = torch.empty_like(x)
    _unrotate_norm_kernel[(rows, hidden // group_size)](
        x,
        gate,
        weight,
        signs,
        out,
        x.stride(-2) if x.dim() > 1 else hidden,
        gate.stride(-2) if gate.dim() > 1 else hidden,
        out.stride(-2) if out.dim() > 1 else hidden,
        n_groups=hidden // group_size,
        group_size=group_size,
        head_dim=head_dim if signs is not None else 1,
        eps=eps,
        HAS_WEIGHT=weight is not None,
        ROTATE=signs is not None,
        num_warps=8,
    )
    return out
