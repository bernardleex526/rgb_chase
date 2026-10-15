from typing import Optional

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


class TargetGallery:
    """Appearance memory of the locked person.

    A fixed anchor (taken at lock time) prevents drift onto another person, while an
    EMA prototype plus a small ring buffer adapts to pose/lighting change.
    """

    def __init__(self, ema_alpha=0.1, max_samples=20):
        self.ema_alpha = ema_alpha
        self.max_samples = max_samples
        self.clear()

    def clear(self):
        self._anchor: Optional[np.ndarray] = None
        self._ema: Optional[np.ndarray] = None
        self._samples = []

    @property
    def empty(self) -> bool:
        return self._anchor is None

    def add(self, feature: np.ndarray):
        f = _normalize(np.asarray(feature, dtype=float))
        if self._anchor is None:
            self._anchor = f
            self._ema = f.copy()
        else:
            self._ema = _normalize((1 - self.ema_alpha) * self._ema + self.ema_alpha * f)
        self._samples.append(f)
        if len(self._samples) > self.max_samples:
            self._samples.pop(1)  # keep the lock-time sample at index 0

    def similarity(self, feature: np.ndarray) -> float:
        if self.empty:
            return 0.0
        f = _normalize(np.asarray(feature, dtype=float))
        sims = [float(self._ema @ f), float(self._anchor @ f)]
        sims += [float(s @ f) for s in self._samples]
        return max(sims)
