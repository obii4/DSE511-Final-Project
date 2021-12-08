import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from src.data import clean_text
from src.data import dimension_4x
from src.data import train_val_test
from src.features import extraction
from src.data import encode
from sklearn.preprocessing import LabelEncoder

rand_seed = 42

#import data
data = pd.read_csv('~/Desktop/mbti_1.csv')

cleaned = clean_text.clean_mbti(data)

label_encoder = LabelEncoder()
cleaned['type'] = label_encoder.fit(cleaned['type'])

print(label_encoder.transform(cleaned['type']))



