import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

def moving_average(series, window):
    """Calculates the moving average of a time series.

    Args:
        series: A pandas Series representing the time series.
        window: The window size for the moving average.

    Returns:
        A pandas Series with the moving average, or None if input is invalid.
    """
    if not isinstance(series, pd.Series):
        print("Error: Input must be a pandas Series.")
        return None
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer.")
        return None
    return series.rolling(window=window).mean()

def exponential_moving_average(series, window):
    """Calculates the exponential moving average of a time series.

    Args:
        series: A pandas Series representing the time series.
        window: The window size for the EMA (often related to the span).

    Returns:
        A pandas Series with the EMA, or None if input is invalid.
    """
    if not isinstance(series, pd.Series):
        print("Error: Input must be a pandas Series.")
        return None
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer.")
        return None
    return series.ewm(span=window, adjust=False).mean() # adjust=False for consistent results

def relative_strength_index(series, window):
    """Calculates the Relative Strength Index (RSI).

    Args:
        series: A pandas Series of price data.
        window: Lookback period for RSI calculation.

    Returns:
        A pandas Series with the RSI values, or None if input is invalid.
    """
    if not isinstance(series, pd.Series):
        print("Error: Input must be a pandas Series.")
        return None
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer.")
        return None

    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    average_gain = up.rolling(window=window).mean()
    average_loss = down.rolling(window=window).mean()
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def standard_deviation(series, window):
    """Calculates the rolling standard deviation of a time series.

    Args:
        series: A pandas Series representing the time series.
        window: The window size for the rolling standard deviation.

    Returns:
        A pandas Series with the rolling standard deviation, or None if input is invalid.
    """
    if not isinstance(series, pd.Series):
        print("Error: Input must be a pandas Series.")
        return None
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer.")
        return None
    return series.rolling(window=window).std()

def bollinger_bands(series, window, num_std=2):
    """Calculates Bollinger Bands.

    Args:
        series: A pandas Series of price data.
        window: Lookback period for moving average and standard deviation.
        num_std: Number of standard deviations for the bands.

    Returns:
        A DataFrame with 'Middle Band', 'Upper Band', and 'Lower Band', or None if input is invalid.
    """
    if not isinstance(series, pd.Series):
        print("Error: Input must be a pandas Series.")
        return None
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer.")
        return None

    middle_band = series.rolling(window=window).mean()
    std_dev = series.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return pd.DataFrame({'Middle Band': middle_band, 'Upper Band': upper_band,
                         'Lower Band': lower_band})


def macd(series, fast_period=12, slow_period=26, signal_period=9):
    """Calculates the Moving Average Convergence Divergence (MACD).

    Args:
        series: A pandas Series of price data.
        fast_period: Period for the fast EMA.
        slow_period: Period for the slow EMA.
        signal_period: Period for the signal line EMA.

    Returns:
        A DataFrame with 'MACD', 'Signal Line', and 'Histogram', or None if input is invalid.
    """
    if not isinstance(series, pd.Series):
        print("Error: Input must be a pandas Series.")
        return None
    if not isinstance(fast_period, int) or fast_period <= 0:
        print("Error: fast_period must be a positive integer.")
        return None
    if not isinstance(slow_period, int) or slow_period <= 0:
        print("Error: slow_period must be a positive integer.")
        return None
    if not isinstance(signal_period, int) or signal_period <= 0:
        print("Error: signal_period must be a positive integer.")
        return None

    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({'MACD': macd_line, 'Signal Line': signal_line, 'Histogram': histogram})


def stochastic_oscillator(high, low, close, window=14):
    """Calculates the Stochastic Oscillator (%K and %D).

    Args:
        high: A pandas Series of high prices.
        low: A pandas Series of low prices.
        close: A pandas Series of closing prices.
        window: Lookback period.

    Returns:
        A DataFrame with '%K' and '%D', or None if input is invalid.
    """
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
      print("Error: Inputs must be pandas Series.")
      return None
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer.")
        return None

    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=3).mean()  # 3-period moving average of %K
    return pd.DataFrame({'%K': k, '%D': d})

def average_directional_index(high, low, close, window=14):
    """Calculates the Average Directional Index (ADX).

    Args:
        high: pandas Series of high prices.
        low: pandas Series of low prices.
        close: pandas Series of closing prices.
        window: Lookback period.

    Returns:
        pandas Series with ADX values, or None if input is invalid.
    """
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
      print("Error: Inputs must be pandas Series.")
      return None
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer.")
        return None

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr1 = pd.DataFrame(high - low).abs()
    tr2 = pd.DataFrame(high - close.shift(1)).abs()
    tr3 = pd.DataFrame(low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(window).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window).mean() / atr
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window).mean()

    return adx

def rate_of_change(series, period=10):
    """Calculates the Rate of Change (ROC).

    Args:
        series: A pandas Series of price data.
        period: Number of periods to look back.

    Returns:
        A pandas Series with the ROC values, or None if input is invalid.
    """
    if not isinstance(series, pd.Series):
        print("Error: Input must be a pandas Series.")
        return None
    if not isinstance(period, int) or period <= 0:
        print("Error: period must be a positive integer.")
        return None
    delta = series.diff(period)
    roc = (delta / series.shift(period)) * 100
    return roc


def calc_technical_indicators(data):
    """Combine all technical indicators in one function.
    Add a set of technical indicators to a dictionary.

    Args:
        dictionary including pandas Series of prices (open, high, low, close) and volume

    Return:
        depending on the indicator pandas Series or data frame

    """

    columns_data = list(data.columns.values)
    columns_data_len = len(columns_data)

    cols = []
    tickers_downl = []
    all_data = {}

    for i in range(columns_data_len-1):
        cols.append(columns_data[i][0])
        tickers_downl.append(columns_data[i][1])

    cols = list(set(cols))
    tickers_downl = list(set(tickers_downl))
    print(f'list of available tickers {tickers_downl}')

    # initialize the dictionary
    for first_key in tickers_downl:
        all_data[first_key] = {}  # Initialize a dictionary for each first-level key

    for col in cols:
        df = pd.DataFrame(data[col])
        for ticker in tickers_downl:
            ts = df[ticker]
            ts.rename(col, inplace=True)
            all_data[ticker][col] = ts

    for ticker in tickers_downl:
        all_data[ticker]['RSI'] = relative_strength_index(all_data[ticker]['Close'], window=14)
        all_data[ticker]['RSI'].dropna(inplace=True)
        all_data[ticker]['RSI'].rename('RSI', inplace=True)
        all_data[ticker]['MA'] = moving_average(all_data[ticker]['Close'], window=14)
        all_data[ticker]['MA'].dropna(inplace=True)
        all_data[ticker]['MA'].rename('MA', inplace=True)
        all_data[ticker]['EMA'] = exponential_moving_average(all_data[ticker]['Close'], window=14)
        all_data[ticker]['EMA'].dropna(inplace=True)
        all_data[ticker]['EMA'].rename('EMA', inplace=True)
        all_data[ticker]['SD'] = standard_deviation(all_data[ticker]['Close'], window=14)
        all_data[ticker]['SD'].dropna(inplace=True)
        all_data[ticker]['SD'].rename('SD', inplace=True)
        all_data[ticker]['Bollinger_Bands'] = bollinger_bands(all_data[ticker]['Close'], window=14)
        all_data[ticker]['Bollinger_Bands'].dropna(inplace=True)
        # all_data[ticker]['Bollinger_Bands'].rename('Bollinger_Bands', inplace=True)
        all_data[ticker]['MACD'] = macd(all_data[ticker]['Close'])
        all_data[ticker]['MACD'].dropna(inplace=True)
        # all_data[ticker]['MACD'].rename('MACD', inplace=True)
        all_data[ticker]['Stochastic_Oscillator'] = stochastic_oscillator(high=all_data[ticker]['High'],
                                                                          low=all_data[ticker]['Low'],
                                                                          close=all_data[ticker]['Close'], window=14)
        all_data[ticker]['Stochastic_Oscillator'].dropna(inplace=True)
        # all_data[ticker]['Stochastic_Oscillator'].rename('Stochastic_Oscillator', inplace=True)
        all_data[ticker]['ADX'] = average_directional_index(high=all_data[ticker]['High'], low=all_data[ticker]['Low'],
                                                            close=all_data[ticker]['Close'], window=14)
        all_data[ticker]['ADX'].dropna(inplace=True)
        all_data[ticker]['ADX'].rename('ADX', inplace=True)
        all_data[ticker]['ROC'] = rate_of_change(all_data[ticker]['Close'])
        all_data[ticker]['ROC'].dropna(inplace=True)
        all_data[ticker]['ROC'].rename('ROC', inplace=True)
        # Add target
        future_price = all_data[ticker]['Close'].shift(-1)
        current_price = all_data[ticker]['Close']
        target = (future_price > current_price).astype(int)
        target.dropna(inplace=True)
        # target = target.head(-1)  # drop last row as prediction is unknown
        all_data[ticker]['Target'] = target
        all_data[ticker]['Target'].rename('Target', inplace=True)

    return all_data

def norm_data(data):
    '''
    Normalize pandas Series to a vector [0,100]
    :param data: pandas timeseries
    :return: normalized series
    '''
    normalized_data = (data - data.min()) / (data.max() - data.min()) * 100

    return normalized_data

def train_random_forest(data):
    # Preprocess data
    features = data.drop(['Target'], axis=1)
    target = data['Target']

    # Split data using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    for train_index, test_index in tscv.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Train the model
        rf.fit(X_train, y_train)

        # Predict on the test set
        predictions = rf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        print(f"Fold Accuracy: {accuracy:.2f}, ROC AUC: {roc_auc:.2f}")

    return rf
