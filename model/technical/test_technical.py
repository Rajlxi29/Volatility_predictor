import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from model.technical.technical import calculate_technical_indicators

def create_dataset(ticker):
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False
    )

    df.columns = df.columns.get_level_values(0)
    df.reset_index()

    return df

def generate_actual_direction(df):

    df["Future_Returns"] = (df["Close"].shift(-5)/(df["Close"]-1))*100

    condition = [
        df["Future_Returns"] > 1,
        df["Future_Returns"] < -1
    ]

    choices = ["Buy", "Sell"]

    df["Actual"] = np.select(
        condition,
        choices,
        default="Hold"
    )

    return df

def generate_predicted_direction(df):

    df = calculate_technical_indicators(df)
    df["Predicted"] = "Hold"

    buy = (
        (df["Close"] > df["MA_50"]) &
        (df["MA_50"] > df["MA_200"]) &
        (df["RSI"] > 50) &
        (df["MACD"] > df["MACD_Signal"])
    )

    sell = (
        (df["Close"] < df["MA_50"]) &
        (df["MA_50"] < df["MA_200"]) &
        (df["RSI"] < 50) &
        (df["MACD"] < df["MACD_Signal"])
    )

    df.loc[buy, "Predicted"] = "Buy"
    df.loc[sell, "Predicted"] = "Sell"

    return df


def evaluate_technical(ticker):

    df = create_dataset(ticker)

    df = generate_actual_direction(df)

    df = generate_predicted_direction(df)

    # Remove rows with unavailable indicators or future returns
    df = df.dropna(
        subset=[
            "MA_50",
            "MA_200",
            "RSI",
            "MACD",
            "MACD_Signal",
            "Actual"
        ]
    ).copy()

    labels = ["Buy", "Hold", "Sell"]

    y_true = df["Actual"]
    y_pred = df["Predicted"]

    print("\nTECHNICAL MODEL EVALUATION")
    print("--------------------------")

    print("Ticker:", ticker)

    print(
        "Accuracy:",
        accuracy_score(y_true, y_pred)
    )

    print(
        "Macro Precision:",
        precision_score(
            y_true, y_pred,
            labels=labels,
            average="macro",
            zero_division=0
        )
    )

    print(
        "Macro Recall:",
        recall_score(
            y_true, y_pred,
            labels=labels,
            average="macro",
            zero_division=0
        )
    )

    print(
        "Macro F1:",
        f1_score(
            y_true, y_pred,
            labels=labels,
            average="macro",
            zero_division=0
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred,
            labels=labels,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true, y_pred,
            labels=labels
        )
    )

    df.to_csv(
        "technical_evaluation_results.csv",
        index=False
    )

if __name__ == "__main__":
    evaluate_technical("ETERNAL.NS")

