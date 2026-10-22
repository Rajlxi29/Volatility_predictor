from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
import pandas as pd

def datasetcreation():
    df = pd.read_parquet("datasets/train_sentiment.parquet")
    print(df.head())

    df = df.rename(columns=
            {
                "sentence": "text",
                "label": "true_label"
            })

    print("Converting Labels")

    if pd.api.types.is_numeric_dtype(df["true_label"]):
        df["true_label"] = df["true_label"].map({
            0: "neutral",
            1: "positive",
            2: "negative"
        })

    print("Downloading the dataset...")

    df["true_label"] = df["true_label"].str.lower()
    df[["text", "true_label"]].to_csv("sentiments.csv", index=False)

classifier = pipeline(
    "text-classification",
    model = "9mark9/finbert-minilm-sentiment"
)

def generate_sentiment(news):

    result = classifier(
        str(news),
        truncation = True,
        max_lenght = 512
    )[0].get("label", [])

    label = result.lower()
    if label not in ["neutral", "positive", "negative"]:
        raise ValueError("""The value should be neutral, positive, negative""")

    return label

def evaluate_sentiment():
    df = pd.read_csv("sentiments.csv")
    df = df.dropna(subset=["text", "true_label"]).copy()

    df["predicted"] = df["text"].apply(generate_sentiment)
    labels = ["neutral", "negative", "positive"]

    y_true = df["true_label"]
    y_pred = df["predicted"]

    print(f"Accuracy : {accuracy_score(y_true, y_pred)}\n")
    print(f"Macro_Precision : {precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)}\n")
    print(f"Macro_Recall : {recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)}\n")
    print(f"Classificaion_report : {classification_report(y_true, y_pred, labels=labels, zero_division=0)}")


if __name__ == "__main__":
    #datasetcreation()
    evaluate_sentiment()