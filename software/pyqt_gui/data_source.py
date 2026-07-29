from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSource(ABC):
    """Minimal data-source interface used by the UI.

    - get_samples(n): expected to return float32-like data with shape (n, 2)
      and an approximate range of -1..1.

    Optional features such as BLE capture (connect/request_capture) are called
    after checking with hasattr().
    """

    @abstractmethod
    def get_samples(self, n: int) -> np.ndarray:  # shape=(n,2)
        raise NotImplementedError
