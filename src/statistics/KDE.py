from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Final

import numpy as np
import rpy2.robjects.packages as rpackages
import scipy
import statsmodels.api as sm
from rpy2 import robjects

from src.statistics.univariate import InterpolatedQuantileFunction, QuantileFunction
from src.tools.array import ensure_strictly_increasing

ks = rpackages.importr("ks")

provenance = rpackages.importr("provenance")


class BandwidthSelectionMethod(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> float:
        pass


class BandwidthSelectionMeta(type(Enum), type(BandwidthSelectionMethod)):
    pass


class BandwidthSelectionMethods(BandwidthSelectionMethod, Enum, metaclass=BandwidthSelectionMeta):
    hscv = auto()
    botev = auto()

    def fit(self, data: np.ndarray) -> float:
        data_r = robjects.FloatVector(data)

        if self is BandwidthSelectionMethods.hscv:
            bw = ks.hscv(data_r)
        elif self is BandwidthSelectionMethods.botev:
            bw = provenance.botev(data_r)
        else:
            raise NotImplementedError("Unknown bandwidth selection method")

        return float(bw[0])


class AverageBandwidthSelectionMethod(BandwidthSelectionMethod):
    def __init__(self, methods: List[BandwidthSelectionMethod]):
        self._methods = methods

    def fit(self, data: np.ndarray) -> float:
        return float(
            np.mean(
                np.array(
                    list(map(lambda method: method.fit(data), self._methods))
                )
            )
        )


averaged_bandwidth_selection_method: Final = AverageBandwidthSelectionMethod(list(BandwidthSelectionMethods))


class KDE:
    def __init__(self, data:np.ndarray, bw: float = None, support_size: int = 2048):
        self._data = data
        self._bw = bw
        self._model = sm.nonparametric.KDEUnivariate(np.array(self._data)).fit(
            bw=np.array(self._bw), gridsize=support_size, fft=False
        )

    def quantile_function(self, gridsize: int = 1001, lim_sup: float = None, mult_factor: float = 1.01):
        if (lim_sup is None) or (lim_sup < self._model.support[-1]):
            lim_sup = self._model.support[-1] * mult_factor

        support = np.append(self._model.support, lim_sup)
        cdf = np.append(self._model.cdf, 1.0)

        if self._model.support[0] > 0:
            support = np.append(0, support)
            cdf = np.append(0, cdf)

        interpolated_cdf = scipy.interpolate.interp1d(
            x=support, y=cdf, kind="linear", bounds_error=True
        )

        grid = np.linspace(0, support[-1], num=gridsize)

        truncated_cdf = (interpolated_cdf(grid) - interpolated_cdf(0)) / (
                interpolated_cdf(lim_sup) - interpolated_cdf(0)
        )

        x, y = ensure_strictly_increasing(first_array=truncated_cdf, second_array=grid)

        return InterpolatedQuantileFunction(probs=x, quantiles=y)


class QuantileFunctionKDEFitter:
    def __init__(
            self,
            gridsize: int = 1001,
            lim_sup: float = None,
            method: BandwidthSelectionMethod = averaged_bandwidth_selection_method,
            mult_factor: float = 1.01,
    ):
        self._method = method
        self._gridsize = gridsize
        self._lim_sup = lim_sup
        self._mult_factor = mult_factor

    def fit(self, data: np.ndarray) -> QuantileFunction:
        bw = self._method.fit(data=data)
        kde = KDE(data=data, bw=bw)

        quantile_function = kde.quantile_function(
            gridsize=self._gridsize,
            lim_sup=self._lim_sup,
            mult_factor=self._mult_factor,
        )

        return quantile_function
