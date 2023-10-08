# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas_ta.overlap import ma
from pandas_ta.utils import get_drift, get_offset, non_zero_range, verify_series

def tmo(open_, close, tmo_length=None, calc_length=None, smooth_length=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: True Momentum Oscillator (TMO)"""

    # Validate arguments (see defaults for TMO, 14/5/3 or 14/6/5)
    tmo_length = int(tmo_length) if tmo_length and tmo_length > 0 else 14
    main_signal = int(calc_length) if calc_length and calc_length > 0 else 5
    smooth_signal = int(smooth_length) if smooth_length and smooth_length > 0 else 3
    mamode = mamode if isinstance(mamode, str) else "ema"

    # verify timeseries
    open_ = verify_series(open_, int(max(tmo_length, calc_length, smooth_length)))
    close = verify_series(close, int(max(tmo_length, calc_length, smooth_length)))
    drift = get_drift(drift)
    offset = get_offset(offset)
    
    if open_ is None or close is None: return
    
    # Calculate the signum dataset of lenght L from open and close prices,
    # and sum the values. This is equivalent to a convolution with a kernel
    # of width tmo_length made of all 1s. This can be done in Numba with
    # a custom convolution since the uniform kernel and a kernel combination
    # of a uniform all 1s kernel and a EMA kernel is a 3rd kernel.
    #
    signum_values = (close - open).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    tmo = signum_values.rolling(tmo_length).sum()
    smooth_tmo = ma(mamode, tmo, length=calc_length)
    
    # Moving averages of the Smooth TMO and Main Signal respectively
    main_signal = ma(mamode, smooth_tmo, length=smooth_length)
    smooth_signal = ma(mamode, main_signal, length=smooth_length)
    
    # i think we need to deal with offset as well
    if offset != 0:
        tmo = tmo.shift(offset)
        smooth_tmo = smooth_tmo.shift(offset)
        main_signal = main_signal.shift(offset)
        smooth_signal = smooth_signal.shift(offset)
        
    # Deal with NaNs, and fill them with the method provided by fill_method,
    # or if no method provided, the default fills NaNs with zeros.
    if "fillna" in kwargs:
        tmo.fillna(kwargs["fillna"], inplace=True)
        smooth_tmo.fillna(kwargs["fillna"], inplace=True)
        main_signal.fillna(kwargs["fillna"], inplace=True)
        smooth_signal.fillna(kwargs["fillna"], inplace=True)
        
    if "fill_method" in kwargs:
        tmo.fillna(method=kwargs["fill_method"], inplace=True)
        smooth_tmo.fillna(method=kwargs["fill_method"], inplace=True)
        main_signal.fillna(method=kwargs["fill_method"], inplace=True)
        smooth_signal.fillna(method=kwargs["fill_method"], inplace=True)       

    result = pd.DataFrame({
        'TMO': tmo,
        'TMO_Smooth': smooth_tmo,
        'TMO_Main_Signal': main_signal,
        'TMO_Smooth_Signal': smooth_signal
    })
    
    tmo.name = f"TMO_{tmo_length}_{calc_length}_{smooth_length}"
    tmo_smooth.name = f"TMO_Smooth_{tmo_length}_{calc_length}_{smooth_length}"
    tmo_main_signal.name = "TMO_Main_Signal_{tmo_length}_{calc_length}_{smooth_length}"
    tmo_smooth_signal.name = "TMO_Smooth_Signal_{tmo_length}_{calc_length}_{smooth_length}"
    tmo.category = tmo_smooth.category = tmo_main_signal.category = tmo_smooth_signal.category = "momentum"
    
    # Prepare dataframe
    df = DataFrame({
            tmo.name: tmo,
            tmo_smooth.name: smooth_tmo,
            tmo_main_signal.name: main_signal,
            tmo_smooth_signal.name: smooth_signal
            })
    
    df.name = f"TMO_{tmo_length}_{calc_length}_{smooth_length}"

    return result

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
    - Smooth TMO: a moving average of the TMO, defaulting to an exponential moving average with period `calc_length`
    - Main Signal: a moving average of the Smooth TMO signal, further smoothing momentum information, with period `smooth_length`
    - Smooth Signal: a moving average of the Main Signal, providing a final trend smoothing and allowing crossover information to be gathered between the Main Signal and Smooth Signal. This uses the period `smooth_length` as well.

Sources:
    https://www.tradingview.com/script/VRwDppqd-True-Momentum-Oscillator/
    https://www.tradingview.com/script/65vpO7T5-True-Momentum-Oscillator-Universal-Edition/

Calculation:
    Default Inputs:
        tmo_length=14, calc_length=5, smooth_length=3

    EMA = Exponential Moving Average
    OO  = Opening price
    CC  = Closing price

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
