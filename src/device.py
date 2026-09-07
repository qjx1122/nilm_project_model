"""Shared device resolution: auto-prefer GPU when one is detected, else keep the original config-driven behavior.

Rule (per repo policy 2026-09-05):
  1. If a usable CUDA GPU is present -> always prefer it (even when the config says "cpu"/"auto";
     a specific "cuda:i" request is honored when that index exists).
  2. Otherwise -> original logic: "auto"/empty resolves to "cpu"; any explicit string is returned
     as-is (so an explicit "cuda" without a GPU still fails downstream exactly like before).
"""
import torch


def cuda_ready() -> bool:
    try:
        return bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
    except Exception:
        return False


def resolve_device(configured="auto"):
    """Return the torch.device to use; prints the decision once for the run log."""
    avail = cuda_ready()
    if avail:
        n = torch.cuda.device_count()
        s = str(configured or "auto")
        if s.startswith("cuda:") and s.split(":", 1)[1].isdigit() and int(s.split(":", 1)[1]) < n:
            dev = torch.device(s)          # honor a valid explicit index
        else:
            dev = torch.device("cuda")    # GPU takes priority over cpu/auto/any other value
        print(f"[device] GPU detected ({n}x, first={torch.cuda.get_device_name(0)!r}) -> using {dev}")
        return dev
    if configured in (None, "", "auto"):
        print("[device] no usable GPU -> using cpu")
        return torch.device("cpu")
    print(f"[device] no usable GPU -> honoring configured device {configured!r} (original behavior)")
    return torch.device(str(configured))
