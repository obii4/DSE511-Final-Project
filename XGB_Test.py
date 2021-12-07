import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.data import clean_text
from src.data import train_val_test
from src.features import extraction
from src.data import encode

#import data
data = pd.read_csv(r"C:\Users\jaypi\JayLocalGit\DSE_511_temp\mbti_1.csv")
cleaned = clean_text.clean_mbti(data)
data_en = encode.label(cleaned) #labels encoded

#text and labels
all_x = data_en['posts']
all_y = data_en['type']

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(all_x, all_y)

#process raw text into ML compatible features
X_train = extraction.feature_Tfidf(X_train)
X_val = extraction.feature_Tfidf(X_val)
X_test = extraction.feature_Tfidf(X_test)


### XGBoost Classification ###
rand_seed = 42

XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", max_depth=1, eta=1, subsample=1)

t0 = time.time()
XGB.fit(X_train, y_train)
t1 = time.time() # ending time
lsvc_all_train_time = t1-t0

#XGB_score = XGB.score(X_test, y_test)

t0 = time.time()
y_true, y_pred_XGB = y_test, XGB.predict(X_test)
t1 = time.time() # ending time
XGB_all_pred_time = t1-t0

XGB_report = classification_report(y_true, y_pred_XGB, output_dict=True)
df_all_jp = pd.DataFrame(XGB_report)

import numpy as np

y_test = np.asarray(y_test)
misclassified = np.where(y_test != y_pred_XGB)

