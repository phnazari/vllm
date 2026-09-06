# LINQ-STATE: LinQuant packed-int recurrent-state support for the Mamba2 mixer.
#
# Phase 1 (speed-parity semantics with the LinQuant HF harness): the fp32/bf16 ssm_state
# pool stays allocated and authoritative for PREFILL; at the prefill->decode handoff each
# sequence's final state is packed once into side pools (uint8 codes + fp32 per-row scales,
# lazily allocated to the same slot count); DECODE then runs int-in-place via
# ``linquant.kernels.state_int.selective_state_update_int`` and never touches the fp pool.
# Enabled by ``additional_config['linq']['state_bits']`` in {4, 6, 8} (LinqConfig); requires
# ``linquant`` (and its fla dep) on PYTHONPATH. Scale grouping: one fp32 scale per (head, dim) row spanning dstate == 128
# (``pack_state`` numerics — the accuracy campaign's ``int{n}_block`` mode for nemotronh).

import os

import torch

from vllm.model_executor.layers.mamba.linq_config import LinqConfig, current as linq_config  # noqa: F401

_PACK_SEED = __import__("itertools").count(1)  # prefill handoff runs eagerly: a host counter is graph-safe here


def linq_bits() -> int:
    """0 = disabled, else 4/6/8 (validated by LinqConfig)."""
    return linq_config().state_bits


def _layer_salt(mixer) -> int:
    """Layer index from the mixer's prefix ('...layers.N...'): decorrelates the SR noise across layers."""
    import re

    m = re.search(r"layers\.(\d+)", getattr(mixer, "prefix", "") or "")
    assert m, f"LINQ SR: no layer index in mixer prefix {getattr(mixer, 'prefix', None)!r}"
    return int(m.group(1)) + 1


def linq_asym() -> bool:
    """Affine INT8 grid (two fp32 per scales row)."""
    return linq_config().state_asym


def _pools(mixer):
    """Codes/scales pools from the layer's accounted cache state (see get_state_shape).

    Returns None during vLLM's memory-profiling phase, when the layer still holds the
    dummy placeholder cache — callers must no-op then. Side allocations are wrong here:
    vLLM sizes the slot pool to fill the GPU, so anything unaccounted either overflows
    (index-OOB) or OOMs (both observed at the b=128 bench).
    """
    kv = mixer.kv_cache
    if len(kv) < 3 or kv[1].numel() == 0:
        return None
    return kv[1], kv[2]  # conv, codes, scales -- the fp state is prefill scratch


# Launch configs for nemotron's shape, swept with the same cold-L2 recipe as vLLM's own fp
# kernel (scripts/investigate/vllm_baseline_launch_tune.py, H100, 2026-08-21).
_TUNED_VLLM = {
    (8, 1): (8, 2), (8, 2): (16, 4), (8, 4): (8, 1), (8, 8): (8, 1),
    (8, 16): (32, 2), (8, 32): (32, 2), (8, 64): (32, 2), (8, 128): (32, 2),
    (4, 1): (32, 8), (4, 2): (32, 8), (4, 4): (32, 4), (4, 8): (32, 4),
    (4, 16): (64, 4), (4, 32): (64, 4), (4, 64): (64, 4), (4, 128): (64, 4),
}


def _ssu_launch(bits, batch):
    """(block_m, num_warps) for the int kernel: nearest measured batch at or below `batch`; (None, None) = its own table."""
    for b in (128, 64, 32, 16, 8, 4, 2, 1):
        if b <= batch and (bits, b) in _TUNED_VLLM:
            return _TUNED_VLLM[(bits, b)]
    return None, None


def linq_scratch(mixer, n, tail, device):
    """Prefill-only fp32 state buffer ``[n, *tail]``, grown on demand and cached on the mixer -- never part of the page."""
    buf = getattr(mixer, "_linq_scratch_buf", None)
    if buf is None or buf.shape[0] < n:
        buf = torch.empty((n, *tail), dtype=torch.float32, device=device)
        mixer._linq_scratch_buf = buf
    return buf[:n]


@torch.no_grad()
def linq_pack_slots(mixer, state_indices, states):
    """Pack ``states`` (fp [n, H, D, N]; GDN/KDA [n, HV, V, K]) into the slots ``state_indices`` (prefill handoff)."""
    from linquant.kernels.state_int.pack_state_kernel import pack_state_to_slots

    pools = _pools(mixer)
    if pools is None:  # profiling-phase dummy cache
        return
    codes, scales = pools
    pack_state_to_slots(states.contiguous().to(torch.float32), codes, scales, state_indices, linq_config().state_bits,
                        asym=linq_config().state_asym, sr_seed=next(_PACK_SEED) if linq_config().state_sr else None)


@torch.no_grad()
def linq_decode(mixer, x, dt, A, B, C, D, dt_bias, state_indices_in,
                state_indices_out, out, num_accepted_tokens, cu_seqlens, seq_lens=None):
    """Int-state decode step; mirrors the ``selective_state_update`` call it replaces."""
    from linquant.kernels.state_int.selective_state_update_int import selective_state_update_int

    # Phase 1 scope: no spec decode, no mamba prefix caching (src slot object == dst slot
    # object holds exactly in that regime; identity check only — no sync under graph capture).
    assert num_accepted_tokens is None, "LINQ int state: spec decode unsupported"
    assert state_indices_out is None or state_indices_out is state_indices_in, (
        "LINQ int state: mamba_cache_mode must be 'none' (dst slots != src slots)"
    )
    if state_indices_in.dim() == 2:  # block-table form [batch, num_blocks]
        assert state_indices_in.shape[1] == 1, (
            f"LINQ int state: expected one state block per seq, got {tuple(state_indices_in.shape)}"
        )
        state_indices_in = state_indices_in.squeeze(1)
    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    block_m, num_warps = _ssu_launch(linq_config().state_bits, x.shape[0])
    selective_state_update_int(
        codes,
        scales,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_indices_in,
        bits=linq_config().state_bits,
        out=out,
        null_block_id=0,  # vLLM v1 pads decode batches with the reserved null block 0
        sr_seed=seq_lens if linq_config().state_sr else None,  # unsliced: a per-call view costs ~4 us of host time per layer
        asym=linq_config().state_asym,
        sr_salt=_layer_salt(mixer) if linq_config().state_sr else 0,
        block_m=block_m,
        num_warps=num_warps,
    )


@torch.no_grad()
def linq_unpack_slots(mixer, state_indices, out):
    """Dequantize slots into `out` (chunked-prefill continuation reads the int pool)."""
    from linquant.kernels.state_int.pack_state_kernel import unpack_state_from_slots

    codes, scales = _pools(mixer)
    return unpack_state_from_slots(out, codes, scales, state_indices, linq_config().state_bits, asym=linq_config().state_asym)


# Gated DeltaNet (qwen3_next / qwen3_5): value-grouped int state in vLLM's own value-major
# [slots, HV, V, K] layout, so pack_state's per-row grouping is already the right axis.


# (BV, warps, stages) per (bits, batch). bits 8 = the copy of vLLM's packed decode kernel, tuned on
# H100 SXM 2026-09-04 with scripts/investigate/vllm_qwen_launch_tune.py --arms packed_int8
# (was qwen_gdn_decode_vllmk.json); bits 4 = the fla-derived vf kernel, same cold-L2 recipe as the fp arm.
_TUNED_GDN = {
    (8, 1): (128, 2, 2), (8, 2): (8, 1, 3), (8, 4): (16, 1, 1), (8, 8): (16, 1, 3),
    (8, 16): (16, 1, 2), (8, 32): (16, 1, 1), (8, 64): (16, 1, 2), (8, 128): (16, 1, 1),
    (4, 1): (8, 1, 3), (4, 2): (8, 1, 3), (4, 4): (16, 1, 3), (4, 8): (8, 1, 2),
    (4, 16): (16, 1, 2), (4, 32): (16, 1, 2), (4, 64): (16, 1, 2), (4, 128): (16, 1, 2),
}


def _launch_or_none(t):
    """(BV, warps, stages) or None when the table has no entry (the kernel copy then uses vLLM's stock launch)."""
    return None if t[0] is None else t


def _gdn_launch(bits, batch):
    for b in (128, 64, 32, 16, 8, 4, 2, 1):
        if b <= batch and (bits, b) in _TUNED_GDN:
            return _TUNED_GDN[(bits, b)]
    return (None, None, None)


@torch.no_grad()


def linq_gdn_decode_vllm(mixer, q, k, v, a, b, A_log, dt_bias, scale, cu_seqlens, state_indices, seq_lens=None):
    """Int-state GDN decode with the copy of vLLM's own kernel (fused_sigmoid_gating_int).

    Same arguments as vLLM's ``fused_sigmoid_gating_delta_rule_update`` call at the decode
    sites (q/k/v are the rearranged ``[1, T, H, K]`` tensors); returns ``o`` shaped like it.
    """
    from linquant.kernels.state_int.fused_sigmoid_gating_int import fused_sigmoid_gating_delta_rule_update_int

    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    assert linq_config().state_bits == 8, "the vLLM-kernel copy is int8 only"
    return fused_sigmoid_gating_delta_rule_update_int(
        A_log, a, b, dt_bias, q, k, v, codes, scales, state_indices,
        scale=scale, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
        sr_seed=seq_lens if linq_config().state_sr else None,  # unsliced persistent buffer, see linq_decode
        sr_salt=_layer_salt(mixer) if linq_config().state_sr else 0,
        asym=linq_config().state_asym, fast=linq_config().state_fast,
    )


def linq_gdn_decode_packed_vllm(mixer, mixed_qkv, a, b, A_log, dt_bias, scale, state_indices, out, seq_lens=None):
    """Int-state GDN decode-only step with the copy of vLLM's packed kernel (reads [q|k|v] directly)."""
    from linquant.kernels.state_int.fused_sigmoid_gating_int import fused_recurrent_gated_delta_rule_packed_decode_int

    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    assert linq_config().state_bits == 8, "the vLLM-kernel copy is int8 only"
    return fused_recurrent_gated_delta_rule_packed_decode_int(
        mixed_qkv, a, b, A_log, dt_bias, scale, codes, scales, out, state_indices, use_qk_l2norm_in_kernel=True,
        sr_seed=seq_lens if linq_config().state_sr else None, sr_salt=_layer_salt(mixer) if linq_config().state_sr else 0,
        asym=linq_config().state_asym, fast=linq_config().state_fast,
        launch=_launch_or_none(_gdn_launch(linq_config().state_bits, mixed_qkv.shape[0])),  # baked (BV, warps, stages) for the copy; vLLM's stock launch when untuned
    )


# Kimi-Linear (KDA): value-grouped int8 pool in vLLM's own [slots, H, V, K] layout, decode on the
# copy of vLLM's fused_recurrent_kda (rule 2026-09-04: INT state only on copies of vLLM's own kernel).
def linq_kda_decode_vllm(mixer, q, k, v, g, beta, cu_seqlens, state_indices, seq_lens=None):
    """Int-state KDA decode step; same arguments as vLLM's ``fused_recurrent_kda`` call site."""
    from linquant.kernels.state_int.fused_recurrent_kda_vllm_int import fused_recurrent_kda_int

    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    assert linq_config().state_bits == 8, "the vLLM-kernel copy is int8 only"
    return fused_recurrent_kda_int(
        q, k, v, g, beta, None, codes, scales, state_indices,
        cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
        sr_seed=seq_lens if linq_config().state_sr else None,  # unsliced persistent buffer, see linq_decode
        sr_salt=_layer_salt(mixer) if linq_config().state_sr else 0,
        asym=linq_config().state_asym, fast=linq_config().state_fast,
    )


# --- parity probe -------------------------------------------------------------------------
# Does the vLLM decode path inject the same quantization error as the HF fake-quant harness
# that produced the accuracy campaign? The two stacks cannot agree bitwise (different chunk
# scan, different attention backend), so the comparison is made on the fp state handed from
# prefill to decode -- the exact tensor both quantizers consume. Dumped only when
# LinqConfig.dump_state_dir names a directory, and only for the FIRST prefill of each layer (later
# prefills in a served run would overwrite with a different sequence's state).
_DUMPED: set[str] = set()


@torch.no_grad()
def linq_dump_handoff_state(mixer, varlen_states) -> None:
    """Persist ``varlen_states`` (fp [n, H, D, N]) for ``mixer`` to ``linq.dump_state_dir``."""
    d = linq_config().dump_state_dir
    if not d:
        return
    key = getattr(mixer, "prefix", "") or f"layer{len(_DUMPED)}"
    if key in _DUMPED:
        return
    _DUMPED.add(key)
    os.makedirs(d, exist_ok=True)
    torch.save(varlen_states.detach().float().cpu(),
               os.path.join(d, key.replace("/", "_") + ".pt"))
