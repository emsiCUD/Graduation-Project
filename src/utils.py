"""
utils.py — Cross-cutting utilities for the ViHSD project.

Currently exposes:

* :func:`set_seed` — fix Python/NumPy/PyTorch RNG state for reproducibility.
* :func:`get_device` — return the best available torch device, with a
  short summary printed when ``verbose=True``.
* :func:`format_seconds` — pretty ``hh:mm:ss`` formatting for log lines.

Import safely from any module without pulling in heavy deps:
:mod:`torch` is imported lazily inside the functions that need it.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Seed Python ``random``, NumPy and (if installed) PyTorch.

    Also flips ``cudnn`` into deterministic / non-benchmark mode so that
    the same input produces the same output across runs. Note this is
    slightly slower than the default benchmark mode — acceptable trade-off
    for a thesis project where reproducibility matters more than the last
    few % of throughput.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(verbose: bool = True):
    """Return ``torch.device('cuda')`` if available, else ``cpu``.

    When ``verbose=True`` prints a one-line summary including GPU name and
    total VRAM. Returns the device object; raises ``ImportError`` if
    PyTorch is not installed.
    """
    import torch

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vram_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        if verbose:
            print(f"Device: cuda ({name}, {vram_gb:.1f} GB VRAM)")
        return torch.device("cuda")

    if verbose:
        print("Device: cpu (no CUDA GPU available)")
    return torch.device("cpu")


def format_seconds(secs: float) -> str:
    """Format a duration in seconds as ``H:MM:SS`` (or ``M:SS`` if < 1h)."""
    secs = int(round(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Timer:
    """Tiny context manager for ``with Timer('step name') as t: …``.

    ``t.elapsed`` is available after exit; the message is printed
    automatically unless ``silent=True``.
    """

    def __init__(self, label: str = "elapsed", silent: bool = False) -> None:
        self.label = label
        self.silent = silent
        self.elapsed: Optional[float] = None
        self._t0: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._t0 is not None
        self.elapsed = time.perf_counter() - self._t0
        if not self.silent:
            print(f"[{self.label}] {format_seconds(self.elapsed)} ({self.elapsed:.2f}s)")


if __name__ == "__main__":
    set_seed(42)
    print("✓ set_seed(42) applied.")
    with Timer("demo"):
        time.sleep(0.05)
