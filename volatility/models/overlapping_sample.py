
def get_variance_overlapping_adjustment_factor(window: int, len_price_data: int):

    h = window
    T = len_price_data
    n = T - h + 1

    assert h < T/2

    a = (h / n)
    b = ((h*h)-1) / (3*(n*n))

    m = 1 / (1 - a + b)

    return m
