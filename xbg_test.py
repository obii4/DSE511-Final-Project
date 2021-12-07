import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.data import clean_text
from src.data import dimension_4x
from src.data import train_val_test
from src.features import extraction

rand_seed = 42

#import data
data = pd.read_csv('~/Desktop/mbti_1.csv')

cleaned = clean_text.clean_mbti(data)

#split in 4 dimensions
EI, NS, TF, JP = dimension_4x.text_split(cleaned)

#text and labels
EI_x = EI['posts']
EI_y = EI['type']

#process raw text into ML compatible features
X = extraction.feature_Tfidf(EI_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, EI_y)

xgb = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.1, max_depth=15, subsample=1)

t0 = time.time()
xgb.fit(X_train,y_train)
t1 = time.time() # ending time
xgb_train_time_EI = t1-t0

t0 = time.time()
y_true, y_pred_xgb = y_test, xgb.predict(X_test)
t1 = time.time() # ending time
xgb_pred_time = t1-t0

xgb_report = classification_report(y_true, y_pred_xgb, output_dict=True)
df_xgb = pd.DataFrame(xgb_report)
