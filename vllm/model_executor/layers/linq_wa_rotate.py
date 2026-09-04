# LINQ-WA-ROT: the ONLINE Hadamards of LinQuant's W/A-quant pipeline (scripts/investigate/wa_quant.py
# BEST_STACK, linquant/prepare.py): R4 on every MLP down_proj input and R_o on every recurrent
# mixer out_proj input. Their inverses are folded into the exported weights, so such a checkpoint
# is only correct with these switched ON. Flag: LINQ_WA_ROT, comma list of
#   r4   plain R4 (Qwen-3.5)            r4s  sign-flipped R4 H.D, D seeded 1000 + layer index (Nemotron-H)
#   ro   R_o on the recurrent mixers' out_proj (both models)
# Independent of LINQ_ROT (the state-side R_k / R_v of linq_rotate.py).
# Numerics: utils.hadamard_utils.matmul_hadU_cuda in fp32, then cast back -- the same code path
# the HF eval runs (neither venv has fast_hadamard_transform, so both use the torch butterfly).
import importlib
import os
import re
import sys

import torch
from torch import nn


def _hadamard_utils():
    """linquant's utils.hadamard_utils, robust to the engine subprocess resolving `utils` elsewhere."""
    try:
        return importlib.import_module("utils.hadamard_utils")
    except ImportError:
        import linquant  # on PYTHONPATH (the state shim needs it too)

        root = os.path.dirname(os.path.dirname(os.path.abspath(linquant.__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        u = sys.modules.get("utils")
        if u is not None and hasattr(u, "__path__") and os.path.join(root, "utils") not in list(u.__path__):
            u.__path__.append(os.path.join(root, "utils"))
        return importlib.import_module("utils.hadamard_utils")

_WA = {r for r in os.environ.get("LINQ_WA_ROT", "").lower().split(",") if r}
if _WA - {"r4", "r4s", "ro"}:
    raise ValueError(f"LINQ_WA_ROT={os.environ.get('LINQ_WA_ROT')} unsupported (want r4 | r4s, ro)")
if {"r4", "r4s"} <= _WA:
    raise ValueError("LINQ_WA_ROT: r4 and r4s are exclusive")


def wa_rot(kind: str) -> bool:
    """True when 'r4' (either flavour) or 'ro' is enabled."""
    return kind in _WA or (kind == "r4" and "r4s" in _WA)


def layer_index(prefix: str) -> int:
    """`...layers.N...` -> N (the seed index of prepare.py's per-layer R4 sign)."""
    m = re.search(r"layers\.(\d+)", prefix)
    assert m, f"LINQ_WA_ROT: no layer index in prefix {prefix!r}"
    return int(m.group(1))


class OnlineHadamard(nn.Module):
    """x -> H.(D.x) on the last axis (fp32), H the get_hadK Kronecker Hadamard of size n."""

    def __init__(self, n: int, sign: torch.Tensor | None = None):
        super().__init__()
        with torch.device("cpu"):  # hadamard_utils builds on the default device (CUDA under vLLM); keep CPU, move lazily
            had_K, K = _hadamard_utils().get_hadK(n)
        self.K = K
        self.register_buffer("had_K", had_K if had_K is not None else torch.zeros(0), persistent=False)
        self.register_buffer("sign", sign if sign is not None else torch.zeros(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matmul_hadU_cuda = _hadamard_utils().matmul_hadU_cuda
        if self.had_K.numel() and self.had_K.device != x.device:
            self.had_K = self.had_K.to(x.device)
        if self.sign.numel():
            if self.sign.device != x.device:
                self.sign = self.sign.to(x.device)
            x = x * self.sign.to(x.dtype)
        had_K = self.had_K if self.had_K.numel() else None
        return matmul_hadU_cuda(x.float(), had_K, self.K).to(x.dtype)


def r4_sign(n: int, prefix: str) -> torch.Tensor:
    """prepare.py's D for the down_proj at `prefix`: randint(0,2) on a CPU generator seeded 1000 + layer."""
    g = torch.Generator(device="cpu").manual_seed(1000 + layer_index(prefix))
    # device='cpu' explicitly: vLLM builds modules under a CUDA default device, prepare.py draws on CPU
    return (torch.randint(0, 2, (n,), generator=g, device="cpu") * 2 - 1).to(torch.float32)


def _no_tp() -> None:
    from vllm.distributed import get_tensor_model_parallel_world_size

    assert get_tensor_model_parallel_world_size() == 1, "LINQ_WA_ROT: the Hadamard needs the unsharded input (TP=1)"


def make_r4(n: int, prefix: str) -> OnlineHadamard | None:
    """The down_proj-input Hadamard for LINQ_WA_ROT, or None when off."""
    if not wa_rot("r4"):
        return None
    _no_tp()
    return OnlineHadamard(n, r4_sign(n, prefix) if "r4s" in _WA else None)


def make_ro(n: int) -> OnlineHadamard | None:
    """The recurrent-mixer out_proj-input Hadamard for LINQ_WA_ROT, or None when off."""
    if not wa_rot("ro"):
        return None
    _no_tp()
    return OnlineHadamard(n)
