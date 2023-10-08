# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame, Series
from pandas_ta.overlap import ma
from pandas_ta.utils import get_offset, verify_series
from numba import jit


@jit(nopython=True)
def compute_combined_kernel(L, J):
    alpha = 2 / (J + 1)
    k3 = np.array([np.sum([(1 - alpha)**(n - m) for m in range(n+1)]) / L for n in range(L)])
    return k3

@jit(nopython=True)
def fast_convolve(signal, kernel):
    return np.convolve(signal, kernel, 'valid')

@jit(nopython=True)
def fast_momentum(signal, mom_len):
    return signal[mom_len:] - signal[:-mom_len]


def tmo(open_, close, tmo_length=None, calc_length=None, smooth_length=None, compute_momemtum=False, mamode=None, offset=None, **kwargs):
    """Indicator: True Momentum Oscillator (TMO)"""

    # Validate arguments
    tmo_length = int(tmo_length) if tmo_length and tmo_length > 0 else 14
    calc_length = int(calc_length) if calc_length and calc_length > 0 else 5
    smooth_length = int(smooth_length) if smooth_length and smooth_length > 0 else 3
    mamode = mamode if isinstance(mamode, str) else "ema"

    # Verify the time series and get the (integer) offset
    open_ = verify_series(open_, max(tmo_length, calc_length, smooth_length))
    close = verify_series(close, max(tmo_length, calc_length, smooth_length))
    offset = get_offset(offset)

    if open_ is None or close is None: return

    # Most indicators compute only the TMO main signal and its smoothed
    # version with an EMA. The main signal is a EMA of the summation of 
    # the signum of the price delta. The summation can be seen as a 
    # convolution with a uniform kernel of all 1s. Since the main TMO
    # signal is an EMA of length calc_lenght of this summation of
    # tmo_lenght elements, these can be combined into a single
    # convolution. Convolution is linear and associative, as such
    #
    # ma[n] = (x[n]*k1[n])*k2[n] = x[n] * (k1[n]*k2[n]) = x[n] * k3[n]
    #
    # where k3[n] is the the convolution of the uniform all 1s kernel and
    # the initial EMA with period calc_length. We loose the ability to
    # support all the Pandas-TA MAs, unless we implement their combined
    # convolution for all supported types, i.e, SMA, EMA, WMA, ...
    # which might be counterproductive. Still, for the default case
    # this leads to
    #
    # \begin{equation}
    # [n] = \begin{cases} \frac{1}{L} \sum_{m=0}^{L-1} (1-\alpha)^{(n-m)} & 
    # \text{for } n \geq 0 \\ 0 & \text{for } n < 0 \end{cases} \\
    # \text{where} \alpha = \frac{2}{J}
    # \end{equation}
    #
    # With the default being an EMA, the optimization is tempting enough to
    # warrant its special case. If the mamode is overriden, then we fallback
    # to the simple summation and Pandas-TA moving averages.
    # Try to JIT both the k3 convolution and the timeseries convolution.
    #
    signum_values = np.sign(close.values - open_.values)
    kernel = np.ones(tmo_length) if mamode != "ema" else compute_combined_kernel(tmo_length, calc_length)
    conv_result = fast_convolve(signum_values, kernel)

    # the default EMA uses the combined kernel, otherwise we must use Pandas-TA MAs
    main_signal = Series(conv_result) if mamode == "ema" else ma(mamode, Series(conv_result), length=calc_length)
    smooth_signal = ma(mamode, tmo_main_signal, length=smooth_length)
    mom_main = mom_smooth = 0

    # Some indicators compute an auxiliary momentum for the main and smooth signal
    if compute_momentum:
        mom_main = Series(fast_momentum(main_signal.values, smooth_length), index=main_signal.index[smooth_length:])
        mom_smooth = Series(fast_momentum(smooth_signal.values, smooth_length), index=smooth_signal.index[smooth_length:])

    # Apply an offset if you wish to shift the timeseries to compare with
    # other indicators or other data.
    offset = 0
    if offset != 0:
        main_signal = main_signal.shift(offset)
        smooth_signal = smooth_signal.shift(offset)
        mom_main = mom_main.shift(offset)
        mom_smooth = mom_smooth.shift(offset)

    # Deal with NaNs in the time series. Unless you specify a method, it will
    # replace all NaNs by zeroes.
    fill_value = kwargs.get("fillna", None)
    fill_method = kwargs.get("fill_method", None)

    if fill_value is not None:
        main_signal.fillna(fill_value, inplace=True)
        smooth_signal.fillna(fill_value, inplace=True)
        mom_main.fillna(fill_value, inplace=True)
        mom_smooth.fillna(fill_value, inplace=True)

    if fill_method is not None:
        main_signal.fillna(method=fill_method, inplace=True)
        smooth_signal.fillna(method=fill_method, inplace=True)
        mom_main.fillna(fill_value, inplace=True)
        mom_smooth.fillna(fill_value, inplace=True)

    # Construct the final DataFrame
    tmo_category = "momentum"
    params = f"{tmo_length}_{calc_length}_{smooth_length}"

#   f"TMO_{params}": tmo[-len(smooth_tmo):],
    df = DataFrame({
        f"TMO_{params}": main_signal,
        f"TMO_Smooth_{params}": smooth_signal,
        f"TMO_Main_Mom_{params}": mom_main,
        f"TMO_Smooth_Mom_{params}": mom_smooth
    })

    df.name = f"TMO_{params}"
    df.category = tmo_category

    return df


tmo.__doc__ = \
    """True Momentum Oscillator (TMO)

The True Momentum Oscillator (TMO) is a technical indicator that aims to capture the 'true momentum'
underlying the price movement of an asset over a specified time frame. It quantifies the net buying or
selling pressure by applying smoothing techniques to a sum of the signum of the differences between
opening and closing prices over the chosen period. Crossovers between the TMO and its smoothed version
generate signals for potential buying or selling opportunities.

The TMO values are scaled to lie within the range [-100, 100]. Overbought and oversold conditions are
commonly identified when the TMO crosses above 70 or below -70, respectively.

Some implementations color the main TMO signal line based on its position relative to the smoothed line
and may include a histogram to represent the distance between these two lines. Variants of the TMO also
calculate the rate of change (momentum) for both the TMO and its smoothed version within the same period.

Sources:
    https://www.tradingview.com/script/VRwDppqd-True-Momentum-Oscillator/
    https://www.tradingview.com/script/65vpO7T5-True-Momentum-Oscillator-Universal-Edition/
    https://www.tradingview.com/script/o9BQyaA4-True-Momentum-Oscillator/

Calculation:
    Default Inputs: `tmo_length=14, calc_length=5, smooth_length=3`

    EMA = Exponential Moving Average
    Delta = close - open
    Signum = 1 if Delta > 0, 0 if Delta = 0, -1 if Delta < 0
    SUM = Summation of N given values
    MA = EMA(SUM(Delta, tmo_length), calc_length)
    TMO = EMA(MA, smooth_length)
    TMOs = EMA(TMO, smooth_length)

Args:
    open_ (pd.Series): Series of 'open' prices.
    close (pd.Series): Series of 'close' prices.
    tmo_length (int, optional): The period for TMO calculation. Default is 14.
    calc_length (int, optional): The period for the Exponential Moving Average of the smoothed TMO. Default is 5.
    smooth_length (int, optional): The period for the Exponential Moving Averages for the Main and Smoothed TMO signals. Default is 3.
    mamode (str, optional): Specifies the type of moving average. For options, see `help(ta.ma)`. Default is 'ema'.
    offset (int, optional): Number of periods to offset the final result. Default is 0.

Kwargs:
    fillna (value, optional): Value with which to fill missing entries. See `pd.DataFrame.fillna(value)`.
    fill_method (str, optional): Method for filling missing entries.

Returns:
    pd.DataFrame: Dataframe containing the columns [TMO, TMO_Smooth, TMO_Main_Mom, TMO_Smooth_Mom].
"""

