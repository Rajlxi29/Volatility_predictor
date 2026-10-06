import yfinance as yf
import pandas as pd
import numpy as np
import time
import os


# ============================================================
# 1. STOCK LIST
# ============================================================

TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "HINDUNILVR.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "TECHM.NS",
    "ULTRACEMCO.NS",
    "NESTLEIND.NS"
]

MARKET_TICKER = "^CNX100"

START_DATE = "2015-01-01"
END_DATE = None

OUTPUT_FILE = "multi_stock_volatility_training.csv"


# ============================================================
# 2. DOWNLOAD MARKET DATA
# ============================================================

print("Downloading CNX100 market data...")

market = yf.download(
    MARKET_TICKER,
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=False,
    progress=False
)

if market.empty:
    raise ValueError("Unable to download CNX100 data.")

# Handle yfinance MultiIndex
if isinstance(market.columns, pd.MultiIndex):
    market.columns = market.columns.get_level_values(0)

market = market.reset_index()

market["Date"] = pd.to_datetime(market["Date"]).dt.tz_localize(None)

market = market.sort_values("Date").reset_index(drop=True)

# Market close
market["NIFTY_100"] = market["Close"]

# Market daily return
market["Market_Return"] = market["NIFTY_100"].pct_change()

# 20-day market volatility
market["Volatility_100"] = (
    market["Market_Return"]
    .rolling(20)
    .std()
)

market_context = market[
    [
        "Date",
        "NIFTY_100",
        "Volatility_100"
    ]
].copy()


# ============================================================
# 3. FUNCTION TO DOWNLOAD ONE STOCK
# ============================================================

def download_stock(ticker):

    print(f"Downloading {ticker}...")

    try:

        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            print(f"WARNING: No data for {ticker}")
            return None

        # Handle MultiIndex returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Remove timezone if present
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        # Keep only required columns
        required = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        df = df[required].copy()

        df["Ticker"] = ticker

        df = df.sort_values("Date").reset_index(drop=True)

        return df

    except Exception as e:

        print(f"ERROR downloading {ticker}: {e}")

        return None


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Daily Return
    # --------------------------------------------------------

    df["Daily_Return"] = (
        df.groupby("Ticker")["Close"]
        .pct_change()
    )

    df["Daily_Return_%"] = (
        df["Daily_Return"] * 100
    )


    # --------------------------------------------------------
    # Moving Averages
    # --------------------------------------------------------

    df["MA_50"] = (
        df.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.rolling(50).mean()
        )
    )

    df["MA_200"] = (
        df.groupby("Ticker")["Close"]
        .transform(
            lambda x: x.rolling(200).mean()
        )
    )


    # --------------------------------------------------------
    # Rolling Volume
    # --------------------------------------------------------

    df["Rolling Volume"] = (
        df.groupby("Ticker")["Volume"]
        .transform(
            lambda x: x.rolling(20).mean()
        )
    )


    # --------------------------------------------------------
    # Rolling Returns
    # --------------------------------------------------------

    df["Rolling_Return_5D"] = (
        df.groupby("Ticker")["Daily_Return_%"]
        .transform(
            lambda x: x.rolling(5).mean()
        )
    )

    df["Rolling_Return_10D"] = (
        df.groupby("Ticker")["Daily_Return_%"]
        .transform(
            lambda x: x.rolling(10).mean()
        )
    )

    df["Rolling_Return_20D"] = (
        df.groupby("Ticker")["Daily_Return_%"]
        .transform(
            lambda x: x.rolling(20).mean()
        )
    )


    # --------------------------------------------------------
    # High / Low percentage
    # --------------------------------------------------------

    df["High_low_%"] = (
        (df["High"] - df["Low"])
        / df["Close"]
    ) * 100


    # --------------------------------------------------------
    # Open / Close percentage
    # --------------------------------------------------------

    df["Open_close_%"] = (
        (df["Close"] - df["Open"])
        / df["Open"]
    ) * 100


    # --------------------------------------------------------
    # True Range
    # --------------------------------------------------------

    previous_close = (
        df.groupby("Ticker")["Close"]
        .shift(1)
    )

    high_low = (
        df["High"] - df["Low"]
    )

    high_close = (
        df["High"] - previous_close
    ).abs()

    low_close = (
        df["Low"] - previous_close
    ).abs()

    df["TR"] = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)


    # --------------------------------------------------------
    # Lagged Volatility
    # Based on 20-day volatility of Daily_Return_%
    # --------------------------------------------------------

    rolling_volatility = (
        df.groupby("Ticker")["Daily_Return_%"]
        .transform(
            lambda x: x.rolling(20).std()
        )
    )

    df["Lagged_Volatility_1"] = (
        rolling_volatility.groupby(df["Ticker"])
        .shift(1)
    )

    df["Lagged_Volatility_5"] = (
        rolling_volatility.groupby(df["Ticker"])
        .shift(5)
    )

    df["Lagged_Volatility_10"] = (
        rolling_volatility.groupby(df["Ticker"])
        .shift(10)
    )


    # --------------------------------------------------------
    # Price / Moving Average ratios
    # --------------------------------------------------------

    df["MA50_to_Price"] = (
        df["MA_50"] / df["Close"]
    )

    df["MA200_to_Price"] = (
        df["MA_200"] / df["Close"]
    )


    # --------------------------------------------------------
    # Rolling Volatility
    # --------------------------------------------------------

    df["Rolling_Volatility_5D"] = (
        df.groupby("Ticker")["Daily_Return_%"]
        .transform(
            lambda x: x.rolling(5).std()
        )
    )

    df["Rolling_Volatility_10D"] = (
        df.groupby("Ticker")["Daily_Return_%"]
        .transform(
            lambda x: x.rolling(10).std()
        )
    )

    df["Rolling_Volatility_20D"] = (
        df.groupby("Ticker")["Daily_Return_%"]
        .transform(
            lambda x: x.rolling(20).std()
        )
    )


    # ========================================================
    # 5. FUTURE 5-DAY VOLATILITY TARGET
    # ========================================================

    def future_volatility(x):

        return (
            x.iloc[::-1]
            .rolling(5)
            .std()
            .iloc[::-1]
            .shift(-1)
        )

    df["future_shift_5"] = (
        df.groupby("Ticker")["Daily_Return"]
        .transform(future_volatility)
    )


    return df


# ============================================================
# 6. DOWNLOAD ALL STOCKS
# ============================================================

all_stocks = []

for ticker in TICKERS:

    stock = download_stock(ticker)

    if stock is not None:
        all_stocks.append(stock)

    # Small delay to reduce request pressure
    time.sleep(0.5)


if not all_stocks:
    raise ValueError("No stock data was downloaded.")


raw_df = pd.concat(
    all_stocks,
    ignore_index=True
)


# ============================================================
# 7. CREATE FEATURES
# ============================================================

print("\nCreating stock features...")

df = create_features(raw_df)


# ============================================================
# 8. ADD MARKET CONTEXT
# ============================================================

print("Adding CNX100 market context...")

df = df.merge(
    market_context,
    on="Date",
    how="left"
)


# ============================================================
# 9. CLEAN DATA
# ============================================================

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# Remove rows where target doesn't exist
# Last 5 rows of every ticker cannot have future volatility.

df = df.dropna(
    subset=["future_shift_5"]
).reset_index(drop=True)


# ============================================================
# 10. FINAL COLUMN ORDER
# ============================================================

feature_columns = [

    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Ticker",

    "Daily_Return_%",
    "MA_50",
    "MA_200",

    "Rolling Volume",

    "Rolling_Return_5D",
    "Rolling_Return_10D",
    "Rolling_Return_20D",

    "High_low_%",
    "Open_close_%",

    "Lagged_Volatility_1",
    "Lagged_Volatility_5",
    "Lagged_Volatility_10",

    "MA50_to_Price",
    "MA200_to_Price",

    "Rolling_Volatility_5D",
    "Rolling_Volatility_10D",
    "Rolling_Volatility_20D",

    "TR",

    "Volatility_100",
    "NIFTY_100",

    "Daily_Return",

    "future_shift_5"
]


df = df[feature_columns]


# ============================================================
# 11. REMOVE REMAINING INVALID ROWS
# ============================================================

print("\nMissing values before final cleanup:")

print(
    df.isna()
      .sum()
      .sort_values(ascending=False)
      .head(15)
)


df = df.dropna().reset_index(drop=True)


# ============================================================
# 12. SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 13. INFORMATION
# ============================================================

print("\n========================================")
print("DATASET CREATED")
print("========================================")

print("Shape:", df.shape)

print(
    "Number of stocks:",
    df["Ticker"].nunique()
)

print(
    "Stocks:",
    df["Ticker"].unique()
)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)

print("\nRows per stock:")

print(
    df["Ticker"]
    .value_counts()
)

print("\nFinal columns:")

print(
    df.columns.tolist()
)

print("\nSaved to:")

print(
    os.path.abspath(OUTPUT_FILE)
)