import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score

def get_sentiment_score(Ytrue, Ypred):

    return round(
        f1_score(Ypred, Ytrue, average="weighted", zero_division=0)*100, 2)

def get_volatility_score(Ytrue, Ypred, scale):

    if scale <= 0:
        raise ValueError("The scale is inappropriate")

    mae = np.mean(np.abs(np.asarray(Ypred) - np.asarray(Ytrue)))
    return round(max(0, (100*(1-(mae/scale)))), 2)

def get_technical_score(Ytrue, Ypred):
    return round(accuracy_score(Ypred, Ytrue)*100,2)