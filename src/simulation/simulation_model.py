import numpy as np
import pandas as pd

from src.statistics.KDE import QuantileFunctionKDEFitter, BandwidthSelectionMethod, averaged_bandwidth_selection_method


class SimulationModel:
    def __init__(
            self, data: np.ndarray,
            gridsize: int = 1001,
            lim_sup: float = None,
            method: BandwidthSelectionMethod = averaged_bandwidth_selection_method,
            mult_factor: float = 1.01,
    ):
        self._data = data
        self._gridsize = gridsize
        self._lim_sup = lim_sup
        self._method = method
        self._mult_factor = mult_factor

        self._quantile_function = QuantileFunctionKDEFitter(
            gridsize=self._gridsize, lim_sup=self._lim_sup, method=self._method, mult_factor=self._mult_factor
        ).fit(data=self._data)

    def sample(
            self,
            num_simulation: int,
            rand: np.random.RandomState = None,
    ) -> np.ndarray:
        if rand is None:
            rand = np.random.RandomState()

        sample = []
        for i in range(0, num_simulation):
            sample = np.array(sample, self._quantile_function(rand.random()))

        return sample


def describe(sample: np.ndarray):
    data = [sample.size]
    data = np.append(data, sample.mean())
    data = np.append(data, sample.std())
    data = np.append(data, sample.min())
    data = np.append(data, np.percentile(sample, [25, 50, 75, 80, 90, 99, 99.50]))
    data = np.append(data, sample.max())
    df = pd.DataFrame(
                data=data,
                columns=['count, mean, std, min, 25%, 50%, 50%, 80%, 90%, 99%, 99.50%, max'],
            )

    return df

