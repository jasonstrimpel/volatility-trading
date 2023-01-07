from volatility.models import GarmanKlass
from volatility.models import HodgesTompkins
from volatility.models import Kurtosis
from volatility.models import Parkinson
from volatility.models import Raw
from volatility.models import RogersSatchell
from volatility.models import Skew
from volatility.models import YangZhang

__all__ = [
    'GarmanKlass',
    'HodgesTompkins',
    'Kurtosis',
    'Parkinson',
    'Raw',
    'RogersSatchell',
    'Skew',
    'YangZhang',
]

def get_variance_overlapping_adjustment_factor(window: int, len_price_data: int):

    h = window
    T = len_price_data
    n = T - h + 1

    assert h < T/2

    a = (h / n)
    b = ((h*h)-1) / (3*(n*n))

    m = 1 / (1 - a + b)

    return m
