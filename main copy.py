import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from src.data import clean_text
from src.data import dimension_4x
from src.features import extraction
from src.data import encode

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
### LogisticRegression ###
lg = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)

t0 = time.time()
lg.fit(X_train,y_train)
t1 = time.time() # ending time
lg_ns_train_time = t1-t0

t0 = time.time()
y_true, y_pred_lg = y_val, lg.predict(X_val)
t1 = time.time() # ending time
lg_ns_pred_time = t1-t0

lg_report = classification_report(y_true, y_pred_lg, output_dict=True)
df_lg_ns = pd.DataFrame(lg_report)

### Linear SVM ###
rand_seed = 42

l_svc = LinearSVC(random_state = 0, C = 10, penalty = 'l2')

t0 = time.time()
l_svc.fit(X_train,y_train)
t1 = time.time() # ending time
lsvc_all_train_time = t1-t0

l_svc_score = l_svc.score(X_test,y_test)

t0 = time.time()
y_true, y_pred_lSVC = y_test, l_svc.predict(X_test)
t1 = time.time() # ending time
lsvc_all_pred_time = t1-t0

l_svc_report = classification_report(y_true, y_pred_lSVC, output_dict=True)
df_all_jp = pd.DataFrame(l_svc_report)








###
###
### This section was evaulate the models when using n = 2 classification ###
##split in 4 dimensions
EI, NS, TF, JP = dimension_4x.text_split(cleaned)

##MBTI Dimensions text and labels
#Extraversion (E) / Introversion (I)
EI_x = EI['posts']
EI_y = EI['type']
EI_ex = extraction.feature_Tfidf(EI_x)
X_train_EI, X_val, X_test, y_train, y_val, y_test = train_val_test.split(EI_ex, EI_y)

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
X = extraction.feature_Tfidf(EI_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, EI_y)

lg = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
t0 = time.time()
lg.fit(X_train,y_train)
t1 = time.time() # ending time
lg_ei_train_time = t1-t0

t0 = time.time()
y_true, y_pred_lg = y_test, lg.predict(X_test)
t1 = time.time() # ending time
lg_ei_pred_time = t1-t0

lg_report = classification_report(y_true, y_pred_lg, output_dict=True)
df_lg_ei = pd.DataFrame(lg_report)

## N/S
#process raw text into ML compatible features
X = extraction.feature_Tfidf(NS_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, NS_y)

lg = LogisticRegression(random_state=0, C=10, penalty='l1', solver = 'liblinear', max_iter=1000)

t0 = time.time()
lg.fit(X_train,y_train)
t1 = time.time() # ending time
lg_ns_train_time = t1-t0

t0 = time.time()
y_true, y_pred_lg = y_test, lg.predict(X_test)
t1 = time.time() # ending time
lg_ns_pred_time = t1-t0

lg_report = classification_report(y_true, y_pred_lg, output_dict=True)
df_lg_ns = pd.DataFrame(lg_report)

## T/F
#process raw text into ML compatible features
X = extraction.feature_Tfidf(TF_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, TF_y)

lg = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)

t0 = time.time()
lg.fit(X_train,y_train)
t1 = time.time() # ending time
lg_tf_train_time = t1-t0

t0 = time.time()
y_true, y_pred_lg = y_test, lg.predict(X_test)
t1 = time.time() # ending time
lg_tf_pred_time = t1-t0

lg_report = classification_report(y_true, y_pred_lg, output_dict=True)
df_lg_tf = pd.DataFrame(lg_report)

## J/P
#process raw text into ML compatible features
X = extraction.feature_Tfidf(JP_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, JP_y)

lg = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)

t0 = time.time()
lg.fit(X_train,y_train)
t1 = time.time() # ending time
lg_jp_train_time = t1-t0

t0 = time.time()
y_true, y_pred_lg = y_test, lg.predict(X_test)
t1 = time.time() # ending time
lg_jp_pred_time = t1-t0

lg_report = classification_report(y_true, y_pred_lg, output_dict=True)
df_lg_JP = pd.DataFrame(lg_report)


### Linear SVM ###
## E/I
#process raw text into ML compatible features
X = extraction.feature_Tfidf(EI_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, EI_y)

l_svc = LinearSVC(random_state = 0, C = 0.0001, penalty = 'l2')

t0 = time.time()
l_svc.fit(X_train,y_train)
t1 = time.time() # ending time
lsvc_ei_train_time = t1-t0

l_svc_score = l_svc.score(X_test,y_test)

t0 = time.time()
y_true, y_pred_lSVC = y_test, l_svc.predict(X_test)
t1 = time.time() # ending time
lsvc_ei_pred_time = t1-t0

l_svc_report = classification_report(y_true, y_pred_lSVC, output_dict=True)
df_lsvc_ei = pd.DataFrame(l_svc_report)

## N/S
#process raw text into ML compatible features
X = extraction.feature_Tfidf(NS_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, NS_y)

l_svc = LinearSVC(random_state = 0, C = 100, penalty = 'l2')

t0 = time.time()
l_svc.fit(X_train,y_train)
t1 = time.time() # ending time
lsvc_ns_train_time = t1-t0

l_svc_score = l_svc.score(X_test,y_test)

t0 = time.time()
y_true, y_pred_lSVC = y_test, l_svc.predict(X_test)
t1 = time.time() # ending time
lsvc_ns_pred_time = t1-t0

l_svc_report = classification_report(y_true, y_pred_lSVC, output_dict=True)
df_lsvc_ns = pd.DataFrame(l_svc_report)

## T/F
#process raw text into ML compatible features
X = extraction.feature_Tfidf(TF_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, TF_y)

l_svc = LinearSVC(random_state = 0, C = 10, penalty = 'l2')

t0 = time.time()
l_svc.fit(X_train,y_train)
t1 = time.time() # ending time
lsvc_tf_train_time = t1-t0

l_svc_score = l_svc.score(X_test,y_test)

t0 = time.time()
y_true, y_pred_lSVC = y_test, l_svc.predict(X_test)
t1 = time.time() # ending time
lsvc_tf_pred_time = t1-t0

l_svc_report = classification_report(y_true, y_pred_lSVC, output_dict=True)
df_lsvc_tf = pd.DataFrame(l_svc_report)

## J/P
#process raw text into ML compatible features
X = extraction.feature_Tfidf(JP_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, JP_y)

l_svc = LinearSVC(random_state = 0, C = 10, penalty = 'l2')

t0 = time.time()
l_svc.fit(X_train,y_train)
t1 = time.time() # ending time
lsvc_jp_train_time = t1-t0

l_svc_score = l_svc.score(X_test,y_test)

t0 = time.time()
y_true, y_pred_lSVC = y_test, l_svc.predict(X_test)
t1 = time.time() # ending time
lsvc_jp_pred_time = t1-t0

l_svc_report = classification_report(y_true, y_pred_lSVC, output_dict=True)
df_lsvc_jp = pd.DataFrame(l_svc_report)

### Random Forest ###
## E/I
#process raw text into ML compatible features
X = extraction.feature_Tfidf(EI_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, EI_y)

rf = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2)

t0 = time.time()
rf.fit(X_train, y_train)
t1 = time.time() # ending time
rf_ei_train_time = t1 - t0

t0 = time.time()
y_true, y_pred_rf = y_test, rf.predict(X_test)
t1 = time.time() # ending time
rf_ei_pred_time = t1-t0

rf_report = classification_report(y_true, y_pred_rf, output_dict=True)
df_rf_ei = pd.DataFrame(rf_report)

## N/S
#process raw text into ML compatible features
X = extraction.feature_Tfidf(NS_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, NS_y)

rf = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100)

t0 = time.time()
rf.fit(X_train, y_train)
t1 = time.time() # ending time
rf_ns_train_time = t1 - t0

t0 = time.time()
y_true, y_pred_rf = y_test, rf.predict(X_test)
t1 = time.time() # ending time
rf_ns_pred_time = t1-t0

rf_report = classification_report(y_true, y_pred_rf, output_dict=True)
df_rf_ns = pd.DataFrame(rf_report)

## T/F
#process raw text into ML compatible features
X = extraction.feature_Tfidf(TF_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, TF_y)

rf = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=60, min_samples_leaf=1, min_samples_split=5, n_estimators=1000)

t0 = time.time()
rf.fit(X_train, y_train)
t1 = time.time() # ending time
rf_tf_train_time = t1 - t0

t0 = time.time()
y_true, y_pred_rf = y_test, rf.predict(X_test)
t1 = time.time() # ending time
rf_tf_pred_time = t1-t0

rf_report = classification_report(y_true, y_pred_rf, output_dict=True)
df_rf_tf = pd.DataFrame(rf_report)

## J/P
#process raw text into ML compatible features
X = extraction.feature_Tfidf(JP_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, JP_y)

rf = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=60, min_samples_leaf=1, min_samples_split=5, n_estimators=100)

t0 = time.time()
rf.fit(X_train, y_train)
t1 = time.time() # ending time
rf_jp_train_time = t1 - t0

t0 = time.time()
y_true, y_pred_rf = y_test, rf.predict(X_test)
t1 = time.time() # ending time
rf_jp_pred_time = t1-t0

rf_report = classification_report(y_true, y_pred_rf, output_dict=True)
df_jp_tf = pd.DataFrame(rf_report)