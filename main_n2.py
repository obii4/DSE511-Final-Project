import pandas as pd
import numpy as np
import time
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


###
###
### This section was evaulate the models when using n = 2 classification ###
##split in 4 dimensions
EI, NS, TF, JP = dimension_4x.text_split(cleaned)

##MBTI Dimensions text and labels
#Extraversion (E) / Introversion (I)
EI_x = EI['posts']
EI_y = EI['type']

#Intuition (N) / Sensing (S)
NS_x = NS['posts']
NS_y = NS['type']

#Thinking (T) / Feeling (F)
TF_x = TF['posts']
TF_y = TF['type']

#Judging (J) / Perceiving (P)
JP_x = JP['posts']
JP_y = JP['type']


#process raw text into ML compatible features
X_EI = extraction.feature_Tfidf(EI_x)
X_NS = extraction.feature_Tfidf(NS_x)
X_TF = extraction.feature_Tfidf(TF_x)
X_JP = extraction.feature_Tfidf(JP_x)

#split text data
EI_X_train, __, EI_X_test, EI_y_train, __, EI_y_test = train_val_test.split(X_EI, EI_y)
NS_X_train, __, NS_X_test, NS_y_train, __, NS_y_test = train_val_test.split(X_NS, NS_y)
TF_X_train, __, TF_X_test, TF_y_train, __, TF_y_test = train_val_test.split(X_TF, TF_y)
JP_X_train, __, JP_X_test, JP_y_train, __, JP_y_test = train_val_test.split(X_JP, JP_y)


### EI ###

LG = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
LSVC = LinearSVC(random_state = 0, C = 0.0001, penalty = 'l2')
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.1, max_depth=10, subsample=1)
RF = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2)

### LogisticRegression ###
times_LG, labels_LG, table_LG = model_eval.eval(LG, EI_X_train, EI_y_train, EI_X_test, EI_y_test)
#store results
times_LG.to_pickle("src/models/results/LG_EI_times.pkl")
labels_LG.to_pickle("src/models/results/LG_EI_labels.pkl")
table_LG.to_pickle("src/models/results/LG_EI_class_results.pkl")

### Linear SVM ###
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, EI_X_train, EI_y_train, EI_X_test, EI_y_test)
#store results
times_LSVC.to_pickle("src/models/results/LSVC_EI_times.pkl")
labels_LSVC.to_pickle("src/models/results/LSVC_EI_labels.pkl")
table_LSVC.to_pickle("src/models/results/LSVC_EI_class_results.pkl")

### XGBoost ###
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, EI_X_train, EI_y_train, EI_X_test, EI_y_test)
#store results
times_XGB.to_pickle("src/models/results/XGB_EI_times.pkl")
labels_XGB.to_pickle("src/models/results/XGB_EI_labels.pkl")
table_XGB.to_pickle("src/models/results/XGB_EI_class_results.pkl")

### Random Forest ###
times_RF, labels_RF, table_RF = model_eval.eval(RF, EI_X_train, EI_y_train, EI_X_test, EI_y_test)
#store results
times_RF.to_pickle("src/models/results/RF_EI_times.pkl")
labels_RF.to_pickle("src/models/results/RF_EI_labels.pkl")
table_RF.to_pickle("src/models/results/RF_EI_class_results.pkl")


### NS ###
LG = LogisticRegression(random_state=0, C=10, penalty='l1', solver = 'liblinear', max_iter=1000)
LSVC = LinearSVC(random_state = 0, C = 100, penalty = 'l2')
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.1, max_depth=10, subsample=1)
RF = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100)

### LogisticRegression ###
times_LG, labels_LG, table_LG = model_eval.eval(LG, NS_X_train, NS_y_train, NS_X_test, NS_y_test)
#store results
times_LG.to_pickle("src/models/results/LG_NS_times.pkl")
labels_LG.to_pickle("src/models/results/LG_NS_labels.pkl")
table_LG.to_pickle("src/models/results/LG_NS_class_results.pkl")

### Linear SVM ###
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, NS_X_train, NS_y_train, NS_X_test, NS_y_test)
#store results
times_LSVC.to_pickle("src/models/results/LSVC_NS_times.pkl")
labels_LSVC.to_pickle("src/models/results/LSVC_NS_labels.pkl")
table_LSVC.to_pickle("src/models/results/LSVC_NS_class_results.pkl")

### XGBoost ###
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, NS_X_train, NS_y_train, NS_X_test, NS_y_test)
#store results
times_XGB.to_pickle("src/models/results/XGB_NS_times.pkl")
labels_XGB.to_pickle("src/models/results/XGB_NS_labels.pkl")
table_XGB.to_pickle("src/models/results/XGB_NS_class_results.pkl")

### Random Forest ###
times_RF, labels_RF, table_RF = model_eval.eval(RF, NS_X_train, NS_y_train, NS_X_test, NS_y_test)
#store results
times_RF.to_pickle("src/models/results/RF_NS_times.pkl")
labels_RF.to_pickle("src/models/results/RF_NS_labels.pkl")
table_RF.to_pickle("src/models/results/RF_NS_class_results.pkl")


### TF ###
LG = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
LSVC = LinearSVC(random_state = 0, C = 10, penalty = 'l2')
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.1, max_depth=10, subsample=.5)
RF = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=60, min_samples_leaf=1, min_samples_split=5, n_estimators=1000)


### LogisticRegression ###
times_LG, labels_LG, table_LG = model_eval.eval(LG, TF_X_train, TF_y_train, TF_X_test, TF_y_test)
#store results
times_LG.to_pickle("src/models/results/LG_TF_times.pkl")
labels_LG.to_pickle("src/models/results/LG_TF_labels.pkl")
table_LG.to_pickle("src/models/results/LG_TF_class_results.pkl")

### Linear SVM ###
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, TF_X_train, TF_y_train, TF_X_test, TF_y_test)
#store results
times_LSVC.to_pickle("src/models/results/LSVC_TF_times.pkl")
labels_LSVC.to_pickle("src/models/results/LSVC_TF_labels.pkl")
table_LSVC.to_pickle("src/models/results/LSVC_TF_class_results.pkl")

### XGBoost ###
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, TF_X_train, TF_y_train, TF_X_test, TF_y_test)
#store results
times_XGB.to_pickle("src/models/results/XGB_TF_times.pkl")
labels_XGB.to_pickle("src/models/results/XGB_TF_labels.pkl")
table_XGB.to_pickle("src/models/results/XGB_TF_class_results.pkl")

### Random Forest ###
times_RF, labels_RF, table_RF = model_eval.eval(RF, TF_X_train, TF_y_train, TF_X_test, TF_y_test)
#store results
times_RF.to_pickle("src/models/results/RF_TF_times.pkl")
labels_RF.to_pickle("src/models/results/RF_TF_labels.pkl")
table_RF.to_pickle("src/models/results/RF_TF_class_results.pkl")

### JP ###
LG = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
LSVC = LinearSVC(random_state = 0, C = 10, penalty = 'l2')
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.05, max_depth=10, subsample=1)
RF = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=60, min_samples_leaf=1, min_samples_split=5, n_estimators=100)

### LogisticRegression ###
times_LG, labels_LG, table_LG = model_eval.eval(LG, JP_X_train, JP_y_train, JP_X_test, JP_y_test)
#store results
times_LG.to_pickle("src/models/results/LG_JP_times.pkl")
labels_LG.to_pickle("src/models/results/LG_JP_labels.pkl")
table_LG.to_pickle("src/models/results/LG_JP_class_results.pkl")

### Linear SVM ###
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, JP_X_train, JP_y_train, JP_X_test, JP_y_test)
#store results
times_LSVC.to_pickle("src/models/results/LSVC_JP_times.pkl")
labels_LSVC.to_pickle("src/models/results/LSVC_JP_labels.pkl")
table_LSVC.to_pickle("src/models/results/LSVC_JP_class_results.pkl")

### XGBoost ###
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, JP_X_train, JP_y_train, JP_X_test, JP_y_test)
#store results
times_XGB.to_pickle("src/models/results/XGB_JP_times.pkl")
labels_XGB.to_pickle("src/models/results/XGB_JP_labels.pkl")
table_XGB.to_pickle("src/models/results/XGB_JP_class_results.pkl")

### Random Forest ###
times_RF, labels_RF, table_RF = model_eval.eval(RF, JP_X_train, JP_y_train, JP_X_test, JP_y_test)
#store results
times_RF.to_pickle("src/models/results/RF_JP_times.pkl")
labels_RF.to_pickle("src/models/results/RF_JP_labels.pkl")
table_RF.to_pickle("src/models/results/RF_JP_class_results.pkl")