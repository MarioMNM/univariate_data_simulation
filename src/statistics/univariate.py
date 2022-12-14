from abc import ABC, abstractmethod
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate

UNEQUAL_LENGTHS: Final = "Different number of probabilities and quantiles"
DISORDERED_PROBS: Final = "Disordered probabilities"
DISORDERED_QUANTILES: Final = "Disordered quantiles"
MISSING_ENDPOINTS: Final = "Probability end points not included"


class QuantileFunction(ABC):
    @abstractmethod
    def __call__(self, prob: float) -> float:
        pass

    @abstractmethod
    def plot(self):
        pass


class InterpolatedQuantileFunction(QuantileFunction):
    def __init__(self, probs: np.ndarray, quantiles: np.ndarray):
        if len(probs) != len(quantiles):
            raise ValueError(UNEQUAL_LENGTHS)

        if np.any(probs != np.sort(np.unique(probs))):
            raise ValueError(DISORDERED_PROBS)

        if np.any(quantiles != np.sort(np.unique(quantiles))):
            raise ValueError(DISORDERED_QUANTILES)

        if probs[0] != 0.0 or probs[-1] != 1.0:
            raise ValueError(MISSING_ENDPOINTS)

        self._probs: Final = probs
        self._quantiles: Final = quantiles

        self._build()

    def __eq__(self, other):
        if isinstance(other, InterpolatedQuantileFunction):
            return (np.all(self._quantiles == other._quantiles)) & (
                np.all(self._quantiles == other._quantiles)
            )
        else:
            return False

    def _build(self):
        self._interpolator = scipy.interpolate.interp1d(
            x=self._probs, y=self._quantiles, kind="linear", bounds_error=True
        )

    def __call__(self, prob: float) -> float:
        return self._interpolator(x=prob)

    def plot(self):
        fig = plt.figure(figsize=(12, 5))

        ax = fig.add_subplot(111)
        ax.plot(
            self._probs,
            self._quantiles,
            color="red",
            label="Quantile Function",
        )

        ax.legend(loc="upper left")

        return ax
