# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame, Series
from pandas_ta.overlap import ma
from pandas_ta.utils import get_offset, verify_series


def tmo(open_, close, tmo_length=None, calc_length=None, smooth_length=None, mamode=None, offset=None, **kwargs):
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

    # Calculate the sum of the signum of the price deltas with period L, but
    # this can be eseen as a convolution with a uniform kernel of all 1s, and
    # allows some optimization.
    # Note that the EMA kernels can be combined with the
    # uniform kernel, for (x[n]*k1)*k2 = x[n]*(k1*k2) = x[n]*k3, where
    # k3 is the first kernel convolved by the second.
    delta = close - open_
    signum_values = np.sign(delta.values)
    tmo = Series(np.convolve(signum_values, np.ones(tmo_length), 'valid'))

    # Normalizing to [-100,100] as shown in the second listed reference, for
    # this allows the user to easily define overbought and oversold conditions,
    # typically -80, and 80.
    normalization_factor = 100 / tmo_length
    tmo *= normalization_factor

    # Rather than using a custom convolution function, use Pandas-TA MAs, since
    # the user might pass its own choise of moving averages, but the TMO
    # default is the exponential moving average (EMA).
    smooth_tmo = ma(mamode, tmo, length=calc_length)
    main_signal = ma(mamode, smooth_tmo, length=smooth_length)
    smooth_signal = ma(mamode, main_signal, length=smooth_length)

    # Apply an offset if you wish to shift the timeseries to compare with
    # other indicators or other data.
    if offset != 0:
        tmo = tmo.shift(offset)
        smooth_tmo = smooth_tmo.shift(offset)
        main_signal = main_signal.shift(offset)
        smooth_signal = smooth_signal.shift(offset)

    # Deal with NaNs in the time series. Unless you specify a method, it will
    # replace all NaNs by zeroes.
    fill_value = kwargs.get("fillna", None)
    fill_method = kwargs.get("fill_method", None)

    if fill_value is not None:
        tmo.fillna(fill_value, inxplace=True)
        smooth_tmo.fillna(fill_value, inplace=True)
        main_signal.fillna(fill_value, inplace=True)
        smooth_signal.fillna(fill_value, inplace=True)

    if fill_method is not None:
        tmo.fillna(method=fill_method, inplace=True)
        smooth_tmo.fillna(method=fill_method, inplace=True)
        main_signal.fillna(method=fill_method, inplace=True)
        smooth_signal.fillna(method=fill_method, inplace=True)

    # Construct the final DataFrame
    tmo_category = "momentum"
    params = f"{tmo_length}_{calc_length}_{smooth_length}"

    df = DataFrame({
        f"TMO_{params}": tmo[-len(smooth_tmo):],
        f"TMO_Smooth_{params}": smooth_tmo,
        f"TMO_Main_Signal_{params}": main_signal,
        f"TMO_Smooth_Signal_{params}": smooth_signal
    })

    df.name = f"TMO_{params}"
    df.category = tmo_category

    return df


tmo.__doc__ = \
    """True Momentum Oscillator (TMO)

The True Momentum Oscillator (TMO) is designed to capture the "true momentum"
behind the price action of an asset over a specific period. It measures the net
buying or selling pressure, by evealuating the signum of the opening and
 closing prices. The oscillator smoothens this measure with a moving average,
 in order to offer a second line with actionable signals. These are useful for
 detecting market trends and reversals, besides crossovers.
The output is scaled between [-100,100] to easily identify overbought and
oversold conditions, where these are commonly set at -80 and 80 threshold
lines.
The smoothed TMO line is further smoothed into a Main Signal, and this in turn
into a Smooth Signal, and their crossovers can provide further insights about
the trends, continuation, reversals. Typically, the Main Signal line is shown
as green or red, if above or below the Smooth Signal line, or an histogram
is computing and shaded based on the same criteria.
In short, the following components are provided:

    - TMO: The base momentum indicator summing the signum of open-close deltas
    - Smooth TMO: a moving average of the TMO, defaulting to an exponential
      moving average with period `calc_length`
    - Main Signal: a moving average of the Smooth TMO signal, further smoothing
      momentum information, with period `smooth_length`
    - Smooth Signal: a moving average of the Main Signal, providing a final
      trend smoothing and allowing crossover information to be gathered between
      the Main Signal and Smooth Signal. This uses the period `smooth_length`
      as well.

Sources:
    https://www.tradingview.com/script/VRwDppqd-True-Momentum-Oscillator/
    https://www.tradingview.com/script/65vpO7T5-True-Momentum-Oscillator-Universal-Edition/

Calculation:
    Default Inputs:
        tmo_length=14, calc_length=5, smooth_length=3

    EMA = Exponential Moving Average
    Delta = close - open
    Signum = 1 if Delta > 0, 0 if Delta = 0, -1 if Delta < 0
    SUM = Summation of N given values
    TMO = SUM(Delta, tmo_length)
    TMO_Smooth=EMA(TMO, calc_length)
    TMO_Main_Signal=EMA(TMO_Smooth, smooth_lenght)
    TMO_Smooth_Signal=EMA(TMO_Main_Signal, smooth_length)

Args:
    open_ (pd.Series): Series of 'open' prices.
    close (pd.Series): Series of 'close' prices.
    tmo_length (int): The period for TMO calculation. Default: 14
    calc_length (int): The EMA period for Smooth TMO. Default: 5
    smooth_length (int): The EMA period for Main and Smooth Signals. Default: 3
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: Columns - [TMO, Smooth_TMO, Main_Signal, Smooth_Signal]
"""
