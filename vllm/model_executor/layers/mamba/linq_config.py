# LINQ-STATE: LinQuant configuration for the vendored vLLM.
#
# The one supported way to configure the integer recurrent state and the online W/A rotations:
#
#   vllm serve <model> ... --additional-config '{"linq": {"state_bits": 8, "state_sr": true, "state_asym": true}}'
#   LLM(model, additional_config={"linq": {"state_bits": 8, "state_sr": True, "wa_rot": "r4,ro"}})
#
# ``VllmConfig.additional_config`` is hashed into the compile-cache key (vllm/config/vllm.py), reaches
# every worker process and is printed with the engine config. ``wa_rot`` defaults to the checkpoint's
# own declaration (``linq_wa_rot`` in the export's config.json): a W/A export is only correct with its
# online rotations on, so the export says which; ``additional_config`` still overrides it.
"""``LinqConfig``: frozen, validated, resolved once per process."""

from dataclasses import dataclass, fields, replace

_ROT_KEYS = frozenset({"r4", "r4s", "ro"})


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
        d["wa_rot"] = _rot_set(d.get("wa_rot", ()))
        if "state_bits" in d:
            d["state_bits"] = int(d["state_bits"])
        return cls(**d)

    @classmethod
    def from_vllm_config(cls, vllm_config) -> "LinqConfig":
        """``additional_config["linq"]`` (default: all off); ``wa_rot`` falls back to the checkpoint's ``linq_wa_rot``."""
        raw = getattr(vllm_config, "additional_config", None) if vllm_config is not None else None
        linq = (raw.get("linq") or {}) if isinstance(raw, dict) else {}
        cfg = cls.from_dict(linq)
        if "wa_rot" not in linq:  # an explicit wa_rot (even empty: the negative-control arm) always wins
            hf = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
            stamped = getattr(hf, "linq_wa_rot", None)
            if stamped:
                cfg = replace(cfg, wa_rot=_rot_set(stamped))
        return cfg

    def banner(self) -> str:
        """The one-line description the runners grep for (``LINQ-STATE: ...``)."""
        rot = f", wa_rot={','.join(sorted(self.wa_rot))}" if self.wa_rot else ""  # W/A rows: the FP32-state arm needs it too
        if not self.state_bits:
            return "LINQ-STATE: off" + rot
        return (f"LINQ-STATE: int{self.state_bits} state, {'asymmetric' if self.state_asym else 'symmetric'} grid, "
                f"stochastic rounding {'ON' if self.state_sr else 'OFF'}, {'fast' if self.state_fast else 'exact'} codec" + rot)


def _rot_set(rot) -> frozenset:
    """'r4, ro' | ['r4', 'ro'] -> frozenset({'r4', 'ro'})."""
    if isinstance(rot, str):
        rot = [r for r in rot.lower().replace(" ", "").split(",") if r]
    return frozenset(rot)


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
    """The pinned config; else the active ``VllmConfig`` context (model construction); else all off."""
    if _RESOLVED is None:
        try:
            from vllm.config import get_current_vllm_config_or_none

            vc = get_current_vllm_config_or_none()
        except Exception:  # noqa: BLE001 - outside an engine (tests, HF harness)
            vc = None
        if vc is not None:
            return set_current(vc)
        return LinqConfig()  # not pinned: a later set_current(vllm_config) wins
    return _RESOLVED
