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


def linq_bits() -> int:
    """0 = disabled, else 4/6/8."""
    if _BITS and _BITS not in (4, 6, 8):
        raise ValueError(f"LINQ_STATE_BITS={_BITS} unsupported (want 4, 6 or 8)")
    return _BITS


def _pools(mixer):
    """Codes/scales pools from the layer's accounted cache state (see get_state_shape).

    Returns None during vLLM's memory-profiling phase, when the layer still holds the
    dummy placeholder cache — callers must no-op then. Side allocations are wrong here:
    vLLM sizes the slot pool to fill the GPU, so anything unaccounted either overflows
    (index-OOB) or OOMs (both observed at the b=128 bench).
    """
    kv = mixer.kv_cache
    if len(kv) < 4 or kv[2].numel() == 0:
        return None
    return kv[2], kv[3]


@torch.no_grad()
def linq_pack_slots(mixer, ssm_state, state_indices, states):
    """Pack ``states`` (fp [n, H, D, N]) into the slots ``state_indices`` (prefill handoff)."""
    from linquant.real_state import pack_state

    pools = _pools(mixer)
    if pools is None:  # profiling-phase dummy cache
        return
    codes, scales = pools
    p = pack_state(states.to(torch.float32), _BITS, 128)
    codes[state_indices] = p.codes
    assert p.scales.shape[-1] == 1, p.scales.shape  # one 128-block per row
    scales[state_indices] = p.scales.squeeze(-1)


@torch.no_grad()
def linq_decode(mixer, ssm_state, x, dt, A, B, C, D, dt_bias, state_indices_in,
                state_indices_out, out, num_accepted_tokens, cu_seqlens):
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
    )
