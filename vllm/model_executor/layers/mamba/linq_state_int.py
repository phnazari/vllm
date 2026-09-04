# LINQ-STATE: LinQuant packed-int recurrent-state support for the Mamba2 mixer.
#
# Phase 1 (speed-parity semantics with the LinQuant HF harness): the fp32/bf16 ssm_state
# pool stays allocated and authoritative for PREFILL; at the prefill->decode handoff each
# sequence's final state is packed once into side pools (uint8 codes + fp32 per-row scales,
# lazily allocated to the same slot count); DECODE then runs int-in-place via
# ``linquant.kernels.state_int.selective_state_update_int`` and never touches the fp pool.
# Enabled by ``LINQ_STATE_BITS`` in {4, 6, 8}; requires ``linquant`` (and its fla dep) on
# PYTHONPATH. Scale grouping: one fp32 scale per (head, dim) row spanning dstate == 128
# (``pack_state`` numerics — the accuracy campaign's ``int{n}_block`` mode for nemotronh).

import os

import torch

_BITS = int(os.environ.get("LINQ_STATE_BITS", "0") or 0)
# Stochastic rounding in the decode requant (LINQ_STATE_SR=1). The per-row seed is the
# request's sequence length: it advances every decode step, and it lives in the model
# runner's persistent buffer, so CUDA-graph replays see the new value. (A host-side seed
# captured into the graph would replay the same dither every step, which stalls under decay
# exactly like RTN -- tests/test_state_int_sr.py.)
_SR = os.environ.get("LINQ_STATE_SR") == "1"
_PACK_SEED = __import__("itertools").count(1)  # prefill handoff runs eagerly: a host counter is graph-safe here
# Asymmetric (affine) INT8 grid, LINQ_STATE_ASYM=1: the Nemotron-H recipe ``int8_block_asym_sr``.
# The scales pool then carries two fp32 per row (scale, min); mamba2 only.
_ASYM = os.environ.get("LINQ_STATE_ASYM") == "1"
if _ASYM and _BITS != 8:
    raise ValueError("LINQ_STATE_ASYM=1 needs LINQ_STATE_BITS=8")
if _BITS:
    print(f"LINQ-STATE: int{_BITS} state, {'asymmetric' if _ASYM else 'symmetric'} grid, "
          f"stochastic rounding {'ON' if _SR else 'OFF'}", flush=True)


def linq_bits() -> int:
    """0 = disabled, else 4/6/8."""
    if _BITS and _BITS not in (4, 6, 8):
        raise ValueError(f"LINQ_STATE_BITS={_BITS} unsupported (want 4, 6 or 8)")
    return _BITS


def _layer_salt(mixer) -> int:
    """Layer index from the mixer's prefix ('...layers.N...'): decorrelates the SR noise across layers."""
    import re

    m = re.search(r"layers\.(\d+)", getattr(mixer, "prefix", "") or "")
    assert m, f"LINQ SR: no layer index in mixer prefix {getattr(mixer, 'prefix', None)!r}"
    return int(m.group(1)) + 1


def linq_asym() -> bool:
    """Affine INT8 grid (two fp32 per scales row)."""
    return _ASYM


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


def _pin_launch(mod, bits, batch):
    """Pin the int kernel's launch config; nearest measured batch at or below `batch`."""
    for b in (128, 64, 32, 16, 8, 4, 2, 1):
        if b <= batch and (bits, b) in _TUNED_VLLM:
            mod.BLOCK_SIZE_M_OVERRIDE, mod.NUM_WARPS_OVERRIDE = _TUNED_VLLM[(bits, b)]
            return


@torch.no_grad()
def linq_pack_slots(mixer, state_indices, states):
    """Pack ``states`` (fp [n, H, D, N]) into the slots ``state_indices`` (prefill handoff)."""
    from linquant.kernels.state_int.pack_state_kernel import pack_state_to_slots

    pools = _pools(mixer)
    if pools is None:  # profiling-phase dummy cache
        return
    codes, scales = pools
    pack_state_to_slots(states.contiguous(), codes, scales, state_indices, _BITS, asym=_ASYM,
                        sr_seed=next(_PACK_SEED) if _SR else None)


@torch.no_grad()
def linq_decode(mixer, x, dt, A, B, C, D, dt_bias, state_indices_in,
                state_indices_out, out, num_accepted_tokens, cu_seqlens, seq_lens=None):
    """Int-state decode step; mirrors the ``selective_state_update`` call it replaces."""
    import importlib

    ssu = importlib.import_module("linquant.kernels.state_int.selective_state_update_int")
    selective_state_update_int = ssu.selective_state_update_int

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
    _pin_launch(ssu, _BITS, x.shape[0])
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
        bits=_BITS,
        out=out,
        null_block_id=0,  # vLLM v1 pads decode batches with the reserved null block 0
        sr_seed=seq_lens if _SR else None,  # unsliced: a per-call view costs ~4 us of host time per layer
        asym=_ASYM,
        sr_salt=_layer_salt(mixer) if _SR else 0,
    )


@torch.no_grad()
def linq_unpack_slots(mixer, state_indices, out):
    """Dequantize slots into `out` (chunked-prefill continuation reads the int pool)."""
    from linquant.kernels.state_int.pack_state_kernel import unpack_state_from_slots

    codes, scales = _pools(mixer)
    return unpack_state_from_slots(out, codes, scales, state_indices, _BITS, asym=_ASYM)


# Gated DeltaNet (qwen3_next / qwen3_5): value-grouped int state in vLLM's own value-major
# [slots, HV, V, K] layout, so pack_state's per-row grouping is already the right axis.


@torch.no_grad()
def linq_pack_slots_gdn(mixer, state_indices, states):
    """Pack ``states`` (fp [n, HV, V, K]) into the slots ``state_indices`` (prefill handoff)."""
    from linquant.kernels.state_int.pack_state_kernel import pack_state_to_slots

    pools = _pools(mixer)
    if pools is None:  # profiling-phase dummy cache
        return
    codes, scales = pools
    pack_state_to_slots(states.contiguous().to(torch.float32), codes, scales,
                        state_indices, _BITS, asym=_ASYM, sr_seed=next(_PACK_SEED) if _SR else None)


# (BV, warps, stages) per (bits, batch), same recipe as the fp arm above.
_TUNED_GDN = {
    (8, 1): (16, 4, 2), (8, 2): (8, 1, 3), (8, 4): (8, 1, 3), (8, 8): (8, 1, 2),
    (8, 16): (8, 1, 3), (8, 32): (8, 1, 3), (8, 64): (8, 1, 2), (8, 128): (8, 1, 1),
    (4, 1): (8, 1, 3), (4, 2): (8, 1, 3), (4, 4): (16, 1, 3), (4, 8): (8, 1, 2),
    (4, 16): (16, 1, 2), (4, 32): (16, 1, 2), (4, 64): (16, 1, 2), (4, 128): (16, 1, 2),
}


def _load_tuned_gdn():
    """Override the baked table from the tuner's JSON when LINQ_GDN_TUNED_JSON points at one."""
    path = os.environ.get("LINQ_GDN_TUNED_JSON")
    if not path or not os.path.exists(path):
        return _TUNED_GDN
    import json

    raw = json.load(open(path))
    return {(int(arm[3:]), int(b)): tuple(cfg)
            for arm, d in raw.items() if arm.startswith("int") for b, cfg in d.items()}


_TUNED_GDN = _load_tuned_gdn() if os.environ.get("LINQ_GDN_TUNED_JSON") else _TUNED_GDN


def _gdn_launch(bits, batch):
    for b in (128, 64, 32, 16, 8, 4, 2, 1):
        if b <= batch and (bits, b) in _TUNED_GDN:
            return _TUNED_GDN[(bits, b)]
    return (None, None, None)


@torch.no_grad()
def linq_gdn_decode(mixer, mixed_qkv, a, b, A_log, dt_bias, scale, state_indices, out,
                    H, HV, K, V, seq_lens=None):
    """Int-state GDN decode step; replaces ``fused_recurrent_gated_delta_rule_packed_decode``.

    Reads the packed qkv buffer directly, exactly like the vendor kernel it replaces, so the
    int arm pays no extra split/copy. ``out`` is [B, 1, HV, V].
    """
    from linquant.kernels.state_int.fused_recurrent_gdn_int import fused_recurrent_gdn_int_vf

    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    nb = mixed_qkv.shape[0]
    bv, nw, ns = _gdn_launch(_BITS, nb)
    o, _ = fused_recurrent_gdn_int_vf(
        None,
        None,
        None,
        codes,
        scales,
        g=a.reshape(nb, 1, -1),
        beta=b.reshape(nb, 1, -1),
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        bits=_BITS,
        bv=bv,
        num_warps=nw,
        num_stages=ns,
        state_indices=state_indices,
        null_block_id=0,  # vLLM v1 pads decode batches with the reserved null block 0
        mixed_qkv=mixed_qkv,
        shape=(nb, H, HV, K, V),
        out=out,
        sr_seed=seq_lens if _SR else None,  # unsliced, see linq_decode
        asym=_ASYM,
        sr_salt=_layer_salt(mixer) if _SR else 0,
    )
    return o


_GDN_KERNEL = os.environ.get("LINQ_GDN_KERNEL", "vllm")  # vllm (default, Philipp 2026-09-04): copy of vLLM's fused_sigmoid_gating kernel + INT-STATE; vf: fla-derived


def linq_gdn_kernel() -> str:
    """Which int8 GDN decode kernel the mixer runs (LINQ_GDN_KERNEL=vllm|vf)."""
    assert _GDN_KERNEL in ("vllm", "vf"), _GDN_KERNEL
    return _GDN_KERNEL


def linq_gdn_decode_vllm(mixer, q, k, v, a, b, A_log, dt_bias, scale, cu_seqlens, state_indices, seq_lens=None):
    """Int-state GDN decode with the copy of vLLM's own kernel (fused_sigmoid_gating_int).

    Same arguments as vLLM's ``fused_sigmoid_gating_delta_rule_update`` call at the decode
    sites (q/k/v are the rearranged ``[1, T, H, K]`` tensors); returns ``o`` shaped like it.
    """
    from linquant.kernels.state_int.fused_sigmoid_gating_int import fused_sigmoid_gating_delta_rule_update_int

    pools = _pools(mixer)
    assert pools is not None, "LINQ int state: decode before cache pools are bound"
    codes, scales = pools
    assert _BITS == 8, "the vLLM-kernel copy is int8 only"
    return fused_sigmoid_gating_delta_rule_update_int(
        A_log, a, b, dt_bias, q, k, v, codes, scales, state_indices,
        scale=scale, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
        sr_seed=seq_lens if _SR else None,  # unsliced persistent buffer, see linq_decode
        sr_salt=_layer_salt(mixer) if _SR else 0,
        asym=_ASYM, fast=os.environ.get("LINQ_STATE_FAST", "1") != "0",
    )


# --- parity probe -------------------------------------------------------------------------
# Does the vLLM decode path inject the same quantization error as the HF fake-quant harness
# that produced the accuracy campaign? The two stacks cannot agree bitwise (different chunk
# scan, different attention backend), so the comparison is made on the fp state handed from
# prefill to decode -- the exact tensor both quantizers consume. Dumped only when
# LINQ_DUMP_STATE names a directory, and only for the FIRST prefill of each layer (later
# prefills in a served run would overwrite with a different sequence's state).
_DUMPED: set[str] = set()


@torch.no_grad()
def linq_dump_handoff_state(mixer, varlen_states) -> None:
    """Persist ``varlen_states`` (fp [n, H, D, N]) for ``mixer`` to $LINQ_DUMP_STATE."""
    d = os.environ.get("LINQ_DUMP_STATE")
    if not d:
        return
    key = getattr(mixer, "prefix", "") or f"layer{len(_DUMPED)}"
    if key in _DUMPED:
        return
    _DUMPED.add(key)
    os.makedirs(d, exist_ok=True)
    torch.save(varlen_states.detach().float().cpu(),
               os.path.join(d, key.replace("/", "_") + ".pt"))
