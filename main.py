from functions import *
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import math

# Define the tickers and the time period
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "^VIX", "SPY"]
period = "6mo"  # 6 months

data = yf.download(tickers, period=period)
data = calc_technical_indicators(data)

pd.concat([bollinger_bands(data['AAPL']['Close'], window=14), data['AAPL']['Close']], axis=1)

# Generate trading signals
# 1 = buy, -1 = sell, 0 = hold current position
# Organize all trading signals in a pandas df
all_signals = pd.DataFrame()

# Define trading signals for multiple indicators
# RSI
all_signals['RSI'] = data['AAPL']['RSI'].apply(lambda x: 1 if x > 70 else (-1 if x < 30 else 0))

# Generate signal for the bollinger bands
# define a function for the signal
def generate_bb_signal(data):
    df = pd.concat([data['AAPL']['Close'], data['AAPL']['Bollinger_Bands'], ], axis=1)
    df.dropna(inplace=True)

    # Create a new column 'Signal' initialized to 0 (hold)
    df['Bollinger_Bands'] = 0

    # Buy signal when price touches or goes below the lower band
    df.loc[df['Close'] <= df['Lower Band'], 'Bollinger_Bands'] = 1

    # Sell signal when price touches or goes above the upper band
    df.loc[df['Close'] >= df['Upper Band'], 'Bollinger_Bands'] = -1

    # Keep only the signal column
    df = df[['Bollinger_Bands']]

    return df

generate_bb_signal(data)

# Combine all required features in a data frame
df = pd.concat([data['AAPL']['Close'],
                data['AAPL']['High'],
                data['AAPL']['Low'],
                data['AAPL']['Open'],
                data['AAPL']['RSI'],
                data['AAPL']['Bollinger_Bands'],
                data['AAPL']['MACD'],
                data['AAPL']['Stochastic_Oscillator'],
                data['AAPL']['Target']
                ], axis=1)

# remove nas and normalize the features
df.dropna(inplace=True)
# df = norm_data(df)
# df['Target'] = df['Target']/100
n = 20
# Create a dataframe for the predictions
backtest = pd.DataFrame({
    'Prediction': [],
    'Probability': [],
    'Actual': []
})

for n in range(n, 0, -1):
    df_back = df.copy()
    df_back.drop(df_back.tail(n).index, inplace = True)

    # train the model
    model = train_random_forest(data=df_back)

    # make prediction
    X_new = df_back.drop(['Target'], axis=1).tail(1)
    # Predict the class (0 or 1)
    predictions = model.predict(X_new)
    # Predict probabilities (e.g., probability of going up)
    probabilities = model.predict_proba(X_new)[:, 1]  # Probability of class '1'

    # append
    row = [predictions[0], probabilities[0], df['Target'].tail(n).values[0]]
    backtest.loc[len(backtest)] = row

