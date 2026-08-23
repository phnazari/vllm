# LINQ-ROT: online Hadamard rotations fused into vLLM's own Triton conv, in prefill
# AND decode.
#
# R_k rotates the d_state axis of B and C by the same R, so it cancels in y = S C.
# R_v rotates the head_dim axis of x, so the state becomes R_v S and the output R_v y;
# that is undone inside vLLM's own gated norm. Enable with LINQ_ROT in {rk, rv, both}.

import math
import os

import torch

_ROT = {
    r
    for r in os.environ.get("LINQ_ROT", "").lower().replace("both", "rk,rv").split(",")
    if r
}


def linq_rot(kind: str) -> bool:
    """True when rotation `kind` ('rk' / 'rv') is enabled."""
    if _ROT - {"rk", "rv"}:
        raise ValueError(
            f"LINQ_ROT={os.environ.get('LINQ_ROT')} unsupported (want rk, rv, both)"
        )
    return kind in _ROT


def linq_norm_fused() -> bool:
    """Route the gated norm through our fused kernel, rotated or not.

    Both arms then run the SAME kernel and differ only by the ROTATE constexpr, so a
    rotation measurement is not confounded by also swapping the norm implementation.
    """
    return os.environ.get("LINQ_NORM_FUSED", "") not in ("", "0")


def linq_rot_any() -> bool:
    """True when any rotation is enabled."""
    return bool(_ROT)


def factor(head_dim: int, device):
    """Sign vector for R = diag(signs) @ H_d / sqrt(d), seeded by LINQ_ROT_SEED."""
    seed = int(os.environ.get("LINQ_ROT_SEED", "0")) + head_dim
    g = torch.Generator(device="cpu").manual_seed(seed)
    # device="cpu" is explicit: vLLM sets a cuda default device during init, and a cpu
    # generator against a cuda default device raises
    bits = torch.randint(0, 2, (head_dim,), generator=g, dtype=torch.int8, device="cpu")
    signs = bits * 2 - 1
    return signs.to(device=device, dtype=torch.float32)


def linq_norm_matrix(head_dim):
    """R = diag(signs) @ H_d / sqrt(d), or None when R_v is off.

    Built at construction, on the default device: vLLM builds weights straight onto
    the target device and never sweeps .to(device), and a host-to-device copy at first
    use would land inside CUDA-graph capture, which raises. torch.compile also traces
    the norm once, so an attribute set later is baked out and never runs.
    """
    from linquant.kernels.hadamard import sylvester_hadamard

    if head_dim is None or not linq_rot("rv"):
        return None
    dev = torch.get_default_device()
    signs = factor(head_dim, dev)
    h = sylvester_hadamard(head_dim, dtype=torch.float32, device=dev)
    return signs[:, None] * h / math.sqrt(head_dim)


def mamba2_spec(mixer):
    """Cache (signs, log2dim, rot_d) for a Mamba-2 mixer's conv channels.

    Channel layout is x || B || C. R_v covers x at head_dim granularity; R_k covers B
    and C together at d_state granularity, so both get the identical rotation and it
    cancels in y = S C. rot_d is the uniform-head fast path, one granularity only.
    """
    spec = getattr(mixer, "_linq_conv_spec", None)
    if spec is not None:
        return spec

    from linquant.kernels.conv_butterfly import make_channel_spec

    dim = mixer.conv1d.weight.size(0)
    dev = mixer.conv1d.weight.device
    inter = mixer.intermediate_size // mixer.tp_size
    bc = mixer.n_groups * mixer.ssm_state_size // mixer.tp_size

    segs, dims = [], set()
    if linq_rot("rv"):
        # same seed as the norm's draw, so R^-1 there undoes exactly R here
        segs.append((0, inter, mixer.head_dim, factor(mixer.head_dim, dev)))
        dims.add(mixer.head_dim)
    if linq_rot("rk"):
        sg = factor(mixer.ssm_state_size, dev)
        segs.append((inter, 2 * bc, mixer.ssm_state_size, sg))
        dims.add(mixer.ssm_state_size)

    signs, log2dim = make_channel_spec(segs, dim, dev)
    covers_all = sum(s[1] for s in segs) == dim
    spec = (signs, log2dim, dims.pop() if len(dims) == 1 and covers_all else 0)
    mixer._linq_conv_spec = spec
    return spec


def linq_unrotate_x(x, rmat):
    """Undo the conv rotation: rows go back through R^T, R = diag(signs) @ H_d / sqrt(d).

    The conv emits y_row = x_row @ R (signs first, then the butterfly), so the row-form
    inverse is y_row @ R^T. Until 2026-08-23 the conv applied R^T and this applied R -- a
    consistent pair, so the round trip was exact, but the state was then quantized in the
    R^T basis where |y @ H @ diag(s)| == |y @ H| and the random signs changed no magnitude
    at all. Both sides moved together; they must stay that way.

    Called from inside vLLM's Mixer2RMSNormGated, ahead of the gate and the RMS, so
    everything else that norm does -- dispatch, the TP collectives, the weight
    multiply -- stays vLLM's. ponytail: dense per-head matmul, fuse into the norm
    kernel if a profile says it matters.
    """
    return (x.reshape(-1, rmat.shape[0]) @ rmat.T.to(x.dtype)).reshape(x.shape)
