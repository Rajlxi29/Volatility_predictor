import pandas as pd
import numpy as np
import pickle
import yfinance as yf

def preprocessing(ticker):

    df = yf.download(
        ticker[0],
        period='2y',
        interval='1d',
        auto_adjust=False
    )

    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Ticker"] = "RELAINCE.NS"

    df["Daily_Return_%"] = df["Close"].pct_change()*100
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()
    df = df.drop("Adj Close", axis=1)

    colist = ['Close', 'High', 'Low', 'Open', 'Volume', 'Daily_Return_%']
    for col in colist:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])

    df["NIFTY_100"] = df["Close"].pct_change()
    df["Volatility_100"] = (df["NIFTY_100"].rolling(20).std())

    prev_close = df["Close"].shift(-1)
    df["TR"] = pd.concat(
        [
            df["High"] - df["Close"], 
            (df["High"] - prev_close).abs(), 
            (df["Low"] - prev_close).abs()
        ], axis=1
    ).max(axis=1)

    #Rolling volume
    df["Rolling Volume"] = df.groupby("Ticker")["Volume"].transform(lambda x : x.rolling(20).mean())

    #Rolling return
    df["Rolling_Return_5D"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change(5))
    df["Rolling_Return_10D"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change(10))
    df["Rolling_Return_20D"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change(20))

    #Ranges
    df["High_low_%"] = ((df["High"] - df["Low"])/df["Close"])*100
    df["Open_close_%"] = ((df["Close"] - df["Open"]).abs()/df["Close"])*100

    #Lagged Volatility
    df["Rolling_Volatility_20D"] = df.groupby("Ticker")["Daily_Return_%"].transform(lambda x: x.rolling(20).std())
    df["Lagged_Volatility_1"] = df.groupby("Ticker")["Rolling_Volatility_20D"].transform(lambda x: x.shift(1))
    df["Lagged_Volatility_5"] = df.groupby("Ticker")["Rolling_Volatility_20D"].transform(lambda x: x.shift(5))
    df["Lagged_Volatility_10"] = df.groupby("Ticker")["Rolling_Volatility_20D"].transform(lambda x: x.shift(10))

    #MA_to_Price
    df["MA50_to_Price"] = (df["MA_50"]/df["Close"])
    df["MA200_to_Price"] = (df["MA_200"]/df["Close"])

    #Volatility
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()

    df["Rolling_Volatility_5D"] = df.groupby("Ticker")["Daily_Return"].transform(lambda x: x.rolling(5).std())
    df["Rolling_Volatility_10D"] = df.groupby("Ticker")["Daily_Return"].transform(lambda x: x.rolling(10).std())

    def future_shift(col):
        return col.iloc[::-1].rolling(5).std().iloc[::-1].shift(-1)

    #Final prediction that is to be made
    df["future_shift_5"] = df.groupby("Ticker")["Daily_Return"].transform(future_shift)
    df = df.drop(["NIFTY_100", "Rolling_Volatility_20D", "future_shift_5", "Ticker", "Date", "Daily_Return"], axis = 1)

    return df

if __name__ == '__main__':
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)

    ticker = "RELAINCE.NS"
    df = preprocessing(ticker)
    prediction = model.predict(df)
    print(np.argmax(prediction))