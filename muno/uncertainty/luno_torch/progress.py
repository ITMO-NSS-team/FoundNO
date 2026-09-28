from __future__ import annotations

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

_ENABLED = True


class _NullBar:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def update(self, *a, **k):
        pass

    def close(self, *a, **k):
        pass

    def refresh(self, *a, **k):
        pass

    def set_postfix(self, *a, **k):
        pass


def set_enabled(value: bool):
    """Глобальный выключатель прогресс-баров (--no-progress)."""
    global _ENABLED
    _ENABLED = bool(value)


def enabled() -> bool:
    return _ENABLED and _tqdm is not None


def tqdm(*args, **kwargs):
    kwargs.setdefault("disable", not enabled())
    if _tqdm is None:  # pragma: no cover - среда без tqdm
        return _NullBar()
    return _tqdm(*args, **kwargs)


def write(msg, *args, **kwargs):
    """Печать, совместимая с активными прогресс-барами."""
    if _tqdm is None:
        print(msg)
        return None
    return _tqdm.write(msg, *args, **kwargs)