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


def _pools(mixer, ssm_state):
    """Lazily allocate codes/scales pools sized to the slot pool (outside graph capture)."""
    if getattr(mixer, "_linq_codes", None) is None:
        from linquant.real_state import pack_state

        slots, nheads, dim, dstate = ssm_state.shape
        assert dstate == 128, f"int state kernel requires dstate == 128, got {dstate}"
        p = pack_state(torch.zeros(1, nheads, dim, dstate, device=ssm_state.device), _BITS, 128)
        mixer._linq_codes = torch.zeros(slots, *p.codes.shape[1:], dtype=torch.uint8, device=ssm_state.device)
        mixer._linq_scales = torch.full((slots, *p.scales.shape[1:]), 1e-12, dtype=torch.float32, device=ssm_state.device)
    return mixer._linq_codes, mixer._linq_scales


@torch.no_grad()
def linq_pack_slots(mixer, ssm_state, state_indices, states):
    """Pack ``states`` (fp [n, H, D, N]) into the slots ``state_indices`` (prefill handoff)."""
    from linquant.real_state import pack_state

    codes, scales = _pools(mixer, ssm_state)
    p = pack_state(states.to(torch.float32), _BITS, 128)
    codes[state_indices] = p.codes
    scales[state_indices] = p.scales


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
    codes, scales = _pools(mixer, ssm_state)
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
