from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _stats(logger):
    if logger is None:
        return None
    if not hasattr(logger, "_doublef_perf_total"):
        logger._doublef_perf_total = {}
    if not hasattr(logger, "_doublef_perf_self"):
        logger._doublef_perf_self = {}
    if not hasattr(logger, "_doublef_perf_stack"):
        logger._doublef_perf_stack = []
    return logger._doublef_perf_total, logger._doublef_perf_self, logger._doublef_perf_stack


def add_time(logger, key: str, total_duration: float, self_duration: float) -> None:
    stats = _stats(logger)
    if stats is None:
        return
    total_stats, self_stats, _ = stats
    total_stats[key] = total_stats.get(key, 0.0) + float(total_duration)
    self_stats[key] = self_stats.get(key, 0.0) + float(self_duration)


def _maybe_sync_cuda(logger) -> None:
    if logger is None:
        return
    if not getattr(logger, "_doublef_sync_cuda", False):
        return
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()


@contextmanager
def timed(logger, key: str):
    stats = _stats(logger)
    if stats is None:
        yield
        return
    _, _, stack = stats
    _maybe_sync_cuda(logger)
    start = perf_counter()
    frame = {"key": key, "start": start, "child": 0.0}
    stack.append(frame)
    try:
        yield
    finally:
        _maybe_sync_cuda(logger)
        end = perf_counter()
        stack.pop()
        total_duration = end - start
        self_duration = total_duration - frame["child"]
        add_time(logger, key, total_duration, self_duration)
        if stack:
            stack[-1]["child"] += total_duration


def format_summary(logger, top_n: int = 12, mode: str = "self") -> list[str]:
    stats = getattr(logger, "_doublef_perf_self" if mode == "self" else "_doublef_perf_total", {})
    if not stats:
        return []
    items = sorted(stats.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [f"{key}={value:.2f}s" for key, value in items]


def get_time(logger, key: str, mode: str = "total") -> float:
    stats = getattr(logger, "_doublef_perf_self" if mode == "self" else "_doublef_perf_total", {})
    return float(stats.get(key, 0.0))
