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

./data/ashrae
train_file = os.path.join(data_path, 'raw_data.csv')


#import data
data = pd.read_csv(r"C:\Users\jaypi\JayLocalGit\DSE_511_temp\mbti_1.csv")
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

### Linear SVM ###
LSVC = LinearSVC(random_state=0, C=10, penalty='l2')
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, X_train, y_train, X_test, y_test)

### XGBoost ###
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", max_depth=15, eta=0.05, subsample=1)
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, X_train, y_train, X_test, y_test)

### Random Forest ###
RF = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2)
times_RF, labels_RF, table_RF = model_eval.eval(RF, X_train, y_train, X_test, y_test)




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

### LogisticRegression ###
## E/I
#process raw text into ML compatible features
EI_x = extraction.feature_Tfidf(EI_x)
NS_x = extraction.feature_Tfidf(NS_x)
TF_x = extraction.feature_Tfidf(TF_x)
JP_x = extraction.feature_Tfidf(JP_x)

#split text data
EI_X_train, __, EI_X_test, EI_y_train, __, EI_y_test = train_val_test.split(EI_x, EI_y)
NS_X_train, __, NS_X_test, NS_y_train, __, NS_y_test = train_val_test.split(NS_x, NS_y)
TF_X_train, __, TF_X_test, TF_y_train, __, TF_y_test = train_val_test.split(TF_x, TF_y)
JP_X_train, __, JP_X_test, JP_y_train, __, JP_y_test = train_val_test.split(JP_x, JP_y)


### EI ###

LG = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
LSVC = LinearSVC(random_state=0, C=10, penalty='l2')
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", max_depth=15, eta=0.05, subsample=1)
RF = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2)

times_LG, labels_LG, table_LG = model_eval.eval(LG, EI_X_train, EI_y_train, EI_X_test, EI_y_test)
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, EI_X_train, EI_y_train, EI_X_test, EI_y_test)
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, EI_X_train, EI_y_train, EI_X_test, EI_y_test)
times_RF, labels_RF, table_RF = model_eval.eval(RF, EI_X_train, EI_y_train, EI_X_test, EI_y_test)

### NS ###

times_LG, labels_LG, table_LG = model_eval.eval(LG, NS_X_train, NS_y_train, NS_X_test, NS_y_test)
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, NS_X_train, NS_y_train, NS_X_test, NS_y_test)
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, NS_X_train, NS_y_train, NS_X_test, NS_y_test)
times_RF, labels_RF, table_RF = model_eval.eval(RF, NS_X_train, NS_y_train, NS_X_test, NS_y_test)

### TF ###

times_LG, labels_LG, table_LG = model_eval.eval(LG, TF_X_train, TF_y_train, TF_X_test, TF_y_test)
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, TF_X_train, TF_y_train, TF_X_test, TF_y_test)
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, TF_X_train, TF_y_train, TF_X_test, TF_y_test)
times_RF, labels_RF, table_RF = model_eval.eval(RF, TF_X_train, TF_y_train, TF_X_test, TF_y_test)

### JP ###

times_LG, labels_LG, table_LG = model_eval.eval(LG, JP_X_train, JP_y_train, JP_X_test, JP_y_test)
times_LSVC, labels_LSVC, table_LSVC = model_eval.eval(LSVC, JP_X_train, JP_y_train, JP_X_test, JP_y_test)
times_XGB, labels_XGB, table_XGB = model_eval.eval(XGB, JP_X_train, JP_y_train, JP_X_test, JP_y_test)
times_RF, labels_RF, table_RF = model_eval.eval(RF, JP_X_train, JP_y_train, JP_X_test, JP_y_test)