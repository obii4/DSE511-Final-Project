import pandas as pd
import numpy as np
import time
import pickle
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.data import train_val_test
from src.data import clean_text
from src.data import dimension_4x
from src.features import extraction
from src.data import encode
from src.models import model_eval

#import data
data = pd.read_csv('~/Desktop/mbti_1.csv')
cleaned = clean_text.clean_mbti(data)
data_en = encode.label(cleaned) #labels encoded

#text and labels
all_x = data_en['posts']
all_y = data_en['type']

#process raw text into ML compatible features
X = extraction.feature_Tfidf(all_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, all_y)

### This section was evaulate the models when using n = 2 classification ###

### Logistic Regression ###
LG = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
times_LG, labels_LG, table_LG = model_eval.eval(LG, X_train, y_train, X_test, y_test)
#store results
times_LG.to_pickle("src/models/results/LG_16_times.pkl")
with open("src/models/results/LG_16_labels.pkl", 'wb') as f:
    pickle.dump(labels_LG, f)
table_LG.to_pickle("src/models/results/LG_16_class_results.pkl")

### Linear SVM ###
LSVC = LinearSVC(random_state=0, C=10, penalty='l2')
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, X_train, y_train, X_test, y_test)
#store results
times_LSVC.to_pickle("src/models/results/LSVC_16_times.pkl")
with open("src/models/results/LSVC_16_labels.pkl", 'wb') as f:
    pickle.dump(labels_LSVC, f)
table_LSVC.to_pickle("src/models/results/LSVC_16_class_results.pkl")

### XGBoost ###
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.05, max_depth=5, subsample=.5)
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, X_train, y_train, X_test, y_test)
#store results
times_XGB.to_pickle("src/models/results/XGB_16_times.pkl")
with open("src/models/results/XGB_16_labels.pkl", 'wb') as f:
    pickle.dump(labels_XGB, f)
table_XGB.to_pickle("src/models/results/XGB_16_class_results.pkl")

### Random Forest ###
RF = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100)
times_RF, labels_RF, table_RF = model_eval.eval(RF, X_train, y_train, X_test, y_test)
#store results
times_RF.to_pickle("src/models/results/RF_16_times.pkl")
with open("src/models/results/RF_16_labels.pkl", 'wb') as f:
    pickle.dump(labels_RF, f)
table_RF.to_pickle("src/models/results/RF_16_class_results.pkl")


