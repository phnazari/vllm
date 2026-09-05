# LINQ-STATE: LinQuant configuration for the vendored vLLM.
#
# The one supported way to configure the integer recurrent state and the online W/A rotations:
#
#   vllm serve <model> ... --additional-config '{"linq": {"state_bits": 8, "state_sr": true, "state_asym": true}}'
#   LLM(model, additional_config={"linq": {"state_bits": 8, "state_sr": True, "wa_rot": "r4,ro"}})
#
# ``VllmConfig.additional_config`` is hashed into the compile-cache key (vllm/config/vllm.py), reaches
# every worker process and is printed with the engine config -- none of which held for the LINQ_*
# environment variables this replaces. The env vars are read only as a deprecated fallback when
# ``additional_config`` carries no ``linq`` key (cleanup proposal 2026-09-04, §B).
"""``LinqConfig``: frozen, validated, resolved once per process."""

import os
import warnings
from dataclasses import dataclass, fields

_ROT_KEYS = frozenset({"r4", "r4s", "ro"})
_ENV_KEYS = ("LINQ_STATE_BITS", "LINQ_STATE_SR", "LINQ_STATE_ASYM", "LINQ_STATE_FAST", "LINQ_WA_ROT", "LINQ_DUMP_STATE")


@dataclass(frozen=True)
class LinqConfig:
    """Serving-time LinQuant settings (all off by default)."""

    state_bits: int = 0  # 0 off | 4 | 6 | 8 (the vLLM-kernel copies for GDN / KDA are int8 only)
    state_sr: bool = False  # stochastic rounding at the prefill handoff and in the decode requant
    state_asym: bool = False  # affine INT8 grid (scale, min) per row; requires state_bits == 8
    state_fast: bool = False  # reciprocal-multiply codecs; False = exact division = every locked recipe
    wa_rot: frozenset = frozenset()  # online rotations of the W/A exports: subset of {r4 | r4s, ro}
    dump_state_dir: str | None = None  # parity probe: dump the first prefill handoff state per layer

    def __post_init__(self):
        if self.state_bits not in (0, 4, 6, 8):
            raise ValueError(f"linq.state_bits={self.state_bits!r}: want 0, 4, 6 or 8")
        if self.state_asym and self.state_bits != 8:
            raise ValueError("linq.state_asym needs state_bits == 8")
        if (self.state_sr or self.state_asym or self.state_fast) and not self.state_bits:
            raise ValueError("linq.state_sr / state_asym / state_fast need state_bits > 0")
        bad = set(self.wa_rot) - _ROT_KEYS
        if bad:
            raise ValueError(f"linq.wa_rot: unknown rotations {sorted(bad)}; allowed {sorted(_ROT_KEYS)}")
        if {"r4", "r4s"} <= set(self.wa_rot):
            raise ValueError("linq.wa_rot: r4 and r4s are exclusive")

    @classmethod
    def from_dict(cls, d: dict) -> "LinqConfig":
        """Parse the ``additional_config["linq"]`` object; unknown keys are errors, ``wa_rot`` may be a string."""
        d = dict(d)
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"additional_config.linq: unknown keys {sorted(unknown)}; known {sorted(names)}")
        rot = d.get("wa_rot", ())
        if isinstance(rot, str):
            rot = [r for r in rot.lower().replace(" ", "").split(",") if r]
        d["wa_rot"] = frozenset(rot)
        if "state_bits" in d:
            d["state_bits"] = int(d["state_bits"])
        return cls(**d)

    @classmethod
    def from_env(cls) -> "LinqConfig":
        """Deprecated fallback: the LINQ_* environment variables (same semantics as before 2026-09-05)."""
        bits = int(os.environ.get("LINQ_STATE_BITS", "0") or 0)
        rot = os.environ.get("LINQ_WA_ROT", "")
        return cls(
            state_bits=bits,
            state_sr=bits > 0 and os.environ.get("LINQ_STATE_SR") == "1",
            state_asym=bits > 0 and os.environ.get("LINQ_STATE_ASYM") == "1",
            state_fast=bits > 0 and os.environ.get("LINQ_STATE_FAST") == "1",
            wa_rot=frozenset(r for r in rot.lower().replace(" ", "").split(",") if r),
            dump_state_dir=os.environ.get("LINQ_DUMP_STATE") or None,
        )

    @classmethod
    def from_vllm_config(cls, vllm_config) -> "LinqConfig":
        """``additional_config["linq"]`` when present, else the env fallback (with a one-time warning if any LINQ_* is set)."""
        raw = getattr(vllm_config, "additional_config", None) if vllm_config is not None else None
        if isinstance(raw, dict) and "linq" in raw:
            return cls.from_dict(raw["linq"] or {})
        if any(os.environ.get(k) for k in _ENV_KEYS):
            warnings.warn("LINQ: configuring through LINQ_* environment variables is deprecated; pass "
                          "--additional-config '{\"linq\": {...}}' instead", stacklevel=2)
        return cls.from_env()

    def banner(self) -> str:
        """The one-line description the runners grep for (``LINQ-STATE: ...``)."""
        if not self.state_bits:
            return "LINQ-STATE: off"
        return (f"LINQ-STATE: int{self.state_bits} state, {'asymmetric' if self.state_asym else 'symmetric'} grid, "
                f"stochastic rounding {'ON' if self.state_sr else 'OFF'}, {'fast' if self.state_fast else 'exact'} codec"
                + (f", wa_rot={','.join(sorted(self.wa_rot))}" if self.wa_rot else ""))


# --- process-wide resolved config ------------------------------------------------------------
# The model classes call ``set_current(vllm_config)`` from their ``get_mamba_state_*_from_config``
# hooks (the first LINQ-aware code that sees the VllmConfig); layers call ``current()`` at init.
_RESOLVED: LinqConfig | None = None


def set_current(vllm_config) -> LinqConfig:
    """Resolve from ``vllm_config`` and pin it for the process (idempotent for an equal config)."""
    global _RESOLVED
    cfg = LinqConfig.from_vllm_config(vllm_config)
    if _RESOLVED is None:
        _RESOLVED = cfg
        if cfg.state_bits or cfg.wa_rot:
            print(cfg.banner(), flush=True)
    elif cfg != _RESOLVED:
        raise RuntimeError(f"LINQ: conflicting configs in one process: {_RESOLVED} vs {cfg}")
    return _RESOLVED


def current() -> LinqConfig:
    """The pinned config; else the active ``VllmConfig`` context (model construction); else the env fallback."""
    global _RESOLVED
    if _RESOLVED is None:
        try:
            from vllm.config import get_current_vllm_config_or_none

            vc = get_current_vllm_config_or_none()
        except Exception:  # noqa: BLE001 - outside an engine (tests, HF harness)
            vc = None
        cfg = LinqConfig.from_vllm_config(vc)
        if vc is not None:
            return set_current(vc)
        return cfg  # env fallback, not pinned: a later set_current(vllm_config) wins
    return _RESOLVED
