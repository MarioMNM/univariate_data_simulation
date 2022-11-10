import pytest
from scipy.stats import norm, rv_continuous

from src.statistics.KDE import BandwidthSelectionMethods, averaged_bandwidth_selection_method, BandwidthSelectionMethod, \
    KDE


@pytest.fixture(params=[norm(loc=0, scale=1), norm(loc=1, scale=1)])
def distribution(request):
    return request.param


@pytest.fixture(
    params=[
        BandwidthSelectionMethods.hlscv,
        BandwidthSelectionMethods.hscv,
        BandwidthSelectionMethods.botev,
        averaged_bandwidth_selection_method,
    ]
)
def method(request):
    return request.param


@pytest.mark.parametrize("sample_size", [1000, 100_000])
def test_bandwidth(
    distribution: rv_continuous,
    sample_size: int,
    method: BandwidthSelectionMethod,
):
    sample = distribution.rvs(size=sample_size)
    h = method.fit(data=sample)
    assert h > 0


@pytest.mark.parametrize("sample_size", [1000, 100_000])
def test_kde(
    distribution: rv_continuous,
    sample_size: int,
    method: BandwidthSelectionMethod,
):
    sample = distribution.rvs(size=sample_size)
    h = method.fit(data=sample)
    kde = KDE(data=sample, bw=h)
    assert len(kde._model.density) != 0


# @pytest.mark.parametrize("sample_size", [1000, 10_000])
# def test_quantile_func(
#     sample_size: int,
# ):
#     sample = norm.rvs(size=sample_size)
#     h = BandwidthSelectionMethods.botev.fit(data=sample)
#     kde = KDE(data=sample, bw=h)
#     quantile_func = kde.build_quantile_function(gridsize=1001)
#     assert quantile_func(0) == 0
#
#
# @pytest.mark.parametrize("sample_size", [1000, 10_000])
# def test_prob(
#     sample_size: int,
# ):
#     sample = norm.rvs(size=sample_size)
#     h = BandwidthSelectionMethods.botev.fit(data=sample)
#     kde = KDE(data=sample, bw=h)
#     quantile_func = kde.build_quantile_function(gridsize=1000)
#     res = quantile_func(prob=0.5)
#     assert res >= 0
#
#
# @pytest.mark.parametrize("sample_size", [1000, 10_000])
# def test_write_json(
#     sample_size: int,
# ):
#     sample = norm.rvs(size=sample_size)
#     h = BandwidthSelectionMethods.botev.fit(data=sample)
#     kde = KDE(data=sample, bw=h)
#     quantile_func = kde.build_quantile_function(gridsize=1000)
#     output_json = quantile_func.to_json()
#     assert type(output_json) == str
#
#
# @pytest.mark.parametrize("sample_size", [1000, 10_000])
# def test_interpolated_kde_quantile_fitter(
#     sample_size: int,
# ):
#     sample = norm.rvs(size=sample_size)
#     quantile_func = InterpolatedKDEQuantileFitter(
#         method=BandwidthSelectionMethods.botev, gridsize=1000
#     )
#     interpolated_quantile_func = quantile_func._fit(data=sample)
#     res = interpolated_quantile_func(prob=0.5)
#     assert res >= 0
