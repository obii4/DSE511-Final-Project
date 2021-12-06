import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label(X):
    label_encoder = LabelEncoder()
    X['type'] = label_encoder.fit_transform(X['type'])
    return X