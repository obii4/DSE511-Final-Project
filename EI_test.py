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

### This section was evaulate the models when using n = 2 classification ###
##split in 4 dimensions
EI, NS, TF, JP = dimension_4x.text_split(cleaned)

##MBTI Dimensions text and labels
#Extraversion (E) / Introversion (I)
EI_x = EI['posts']
EI_y = EI['type']
EI_ex = extraction.feature_Tfidf(EI_x)
X_EI_train, X_EI_val, X_EI_test, y_EI_train, y_EI_val, y_EI_test = train_val_test.split(EI_ex, EI_y)


### LogisticRegression ###
## E/I

lg = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
t0 = time.time()
lg.fit(X_EI_train,y_EI_train)
t1 = time.time() # ending time
lg_ei_train_time = t1-t0

t0 = time.time()
y_true, y_pred_EI_lg = y_EI_test, lg.predict(X_EI_test)
t1 = time.time() # ending time
lg_ei_pred_time = t1-t0

lg_report = classification_report(y_true, y_pred_EI_lg, output_dict=True)
df_lg_ei = pd.DataFrame(lg_report)

### Linear SVM ###
## E/I

l_svc = LinearSVC(random_state = 0, C = 0.0001, penalty = 'l2')

t0 = time.time()
l_svc.fit(X_EI_train,y_EI_train)
t1 = time.time() # ending time
lsvc_ei_train_time = t1-t0

l_svc_score = l_svc.score(X_EI_test,y_EI_test)

t0 = time.time()
y_true, y_pred_EI_lSVC = y_test, l_svc.predict(X_EI_test)
t1 = time.time() # ending time
lsvc_ei_pred_time = t1-t0

l_svc_report = classification_report(y_true, y_pred_EI_lSVC, output_dict=True)
df_lsvc_ei = pd.DataFrame(l_svc_report)

### Random Forest ###
## E/I
#process raw text into ML compatible features

rf = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2)

t0 = time.time()
rf.fit(X_EI_train, y_EI_train)
t1 = time.time() # ending time
rf_ei_train_time = t1 - t0

t0 = time.time()
y_true, y_pred_EI_rf = y_test, rf.predict(X_EI_test)
t1 = time.time() # ending time
rf_ei_pred_time = t1-t0

rf_report = classification_report(y_true, y_pred_EI_rf, output_dict=True)
df_rf_ei = pd.DataFrame(rf_report)
