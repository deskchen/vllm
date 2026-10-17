# SPDX-License-Identifier: Apache-2.0
"""Phantora cross-process simulated-time propagation.

vLLM V1 runs the model in a separate EngineCore process. Under Phantora's
simulator each process keeps its *own* virtual clock, so the frontend (which only
waits in real time) and EngineCore (which advances simulated time) drift apart.

We reconcile them causally (Lamport-clock style): every message across the
boundary carries the sender's simulated time (``stamp()``), and the receiver
advances its own clock forward to it (``adopt()`` == a max). Stamping both
directions makes the result the critical path -- non-overlapped work sums,
overlapped work is hidden.

Entirely a no-op unless running under the Phantora simulator (detected via the
``PHANTORA_SOCKET_PREFIX`` env the simulator sets), so normal vLLM is unaffected.
"""

import ctypes
import os

_enabled = False
_stamp = None
_adopt = None


def _init() -> None:
    global _enabled, _stamp, _adopt
    if os.environ.get("PHANTORA_SOCKET_PREFIX") is None:
        return  # not running under the Phantora simulator
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.get_time_double.restype = ctypes.c_double
        lib.phantora_adopt_time_double.argtypes = [ctypes.c_double]
        lib.phantora_adopt_time_double.restype = None
        _stamp = lib.get_time_double
        _adopt = lib.phantora_adopt_time_double
        _enabled = True
    except (OSError, AttributeError):
        pass  # stub not loadable / missing symbol -> stay disabled


_init()


def enabled() -> bool:
    return _enabled


def stamp() -> float:
    """This process's current simulated time in seconds (0.0 if disabled)."""
    return _stamp() if _enabled else 0.0


def adopt(t: float) -> None:
    """Advance this process's virtual clock FORWARD to ``t`` seconds (a max).

    No-op if disabled or ``t`` is unset/0.0. Never moves the clock backward.
    """
    if _enabled and t and t > 0.0:
        _adopt(t)
