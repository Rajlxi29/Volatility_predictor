from config.settings import OPENAI_API, OPENAI_BASE
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal
import pandas as pd
import yfinance as yf
import json



class summary(BaseModel):
    Trend : str = Field(description="value with meaning in one line")
    RSI : str = Field(description="value with meaning in one line")
    MACD : str = Field(description="value with meaning in one line")
    Volume : str = Field(description="value with meaning in one line")
    Bollinger : str = Field(description="value with meaning in one line")
    Summary: str = Field(description="summary after reveiewing the values in 50 words")
    Position: Literal["Hold", "Sell", "Buy"]

def datacollection(ticker):

    result = yf.download(
        ticker[0],
        period="2y",
        interval="1d",
        auto_adjust=False
    )

    result.columns = result.columns.get_level_values(0)
    result = result.reset_index()
    result["Ticker"] = ticker

    return result

def calculate_technical_indicators(df):

    df["MA_50"] = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()

    df["Daily_Return"] = df["Close"].pct_change()
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain/avg_loss
    df["RSI"] = 100 - (100/(1+rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"]=ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()

    df["BB_upper"] = rolling_mean + 2 * rolling_std
    df["BB_lower"] = rolling_mean - 2 * rolling_std

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    return df

def technical_agent(df):

    df = calculate_technical_indicators(df)

    latest = df.iloc[-1]

    signals = {}
    # signals["Price"] = latest["Price"]
    signals["RSI"] = latest["RSI"]
    signals["MACD"] = latest["MACD"]
    signals["MACD_signals"] = latest["MACD_Signal"]
    signals["ATR"] = latest["ATR"]
    signals["Volume_Ratio"] = latest["Volume_Ratio"]

    calculations = {}

    # Trend
    if latest["Close"] > latest["MA_50"]:
        calculations["MA50"] = "above"
    else:
        calculations["MA50"] = "below"

    if latest["Close"] > latest["MA_200"]:
        calculations["MA200"] = "above"
    else:
        calculations["MA200"] = "below"

    # RSI
    if latest["RSI"] > 70:
        calculations["RSI"] = "overbought"
    elif latest["RSI"] < 30:
        calculations["RSI"] = "oversold"
    else:
        calculations["RSI"] = "neutral"

    # MACD
    if latest["MACD"] > latest["MACD_Signal"]:
        calculations["MACD"] = "bullish"
    else:
        calculations["MACD"] = "bearish"

    # Volume
    if latest["Volume_Ratio"] > 1.5:
        calculations["Volume"] = "high"
    elif latest["Volume_Ratio"] < 0.7:
        calculations["Volume"] = "low"
    else:
        calculations["Volume"] = "normal"

    # Bollinger
    if latest["Close"] > latest["BB_upper"]:
        calculations["Bollinger"] = "above_upper_band"
    elif latest["Close"] < latest["BB_lower"]:
        calculations["Bollinger"] = "below_lower_band"
    else:
        calculations["Bollinger"] = "inside_bands"

    signals["calculation"] = calculations

    return signals

def solve(signals) -> str:
    # query = f"""
    #     generate the technical summary of the {signals} data that is provided to you do it
    #     by focusing on the part if It is good to buy, sell or hold the stock the format of the output should be like this
    #     Summary of the signals

    #     Trend : 
    #     RSI : 
    #     MACD : 
    #     Volume : 
    #     Bollinger : 

    #     Complete situation summary and approximat verdict on the position
    #     BUY, HOLD, SELL
    # """

    sys = SystemMessage(content="""You are an expert stock market analyst.
                                Analyze the provided technical signals.
                                Return the result strictly according to the provided structured output schema.
                                Do not return Markdown.
                                Do not add headings or explanations outside the schema.""")
    hum = HumanMessage(content=json.dumps(signals, indent=2))
    query = [sys, hum]

    model = ChatOpenAI(
        model = "openrouter/free",
        openai_api_key = OPENAI_API,
        openai_api_base = OPENAI_BASE,
        timeout=30,
        max_retries=2
    )
    modell = model.with_structured_output(summary)
    try:
        response = modell.invoke(query)
        return response

    except ConnectionError as e:
        print(f"The model is not connected properly : {e}")
        return ""

    except TimeoutError as e:
        print(f"Timeout for retries: {e}")
        return ""

    except Exception as e:
        print(f"Model failed to produce output: {e}")
        return None


def technical_init(ticker: str):
    print("getting the dataset...")
    df = datacollection(ticker)
    print("getting the signals...")
    signal = technical_agent(df)
    print("Done\n")
    print("getting summary...")
    summary = solve(signal)
    return summary

if __name__ == '__main__':
    ticker = "RELAINCE.NS"
    signal = technical_init(ticker)
    print(signal)