import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


from src.data import clean_text
from src.data import dimension_4x
from src.data import train_val_test
from src.features import extraction

#import data
data = pd.read_csv('~/Desktop/mbti_1.csv')

cleaned = clean_text.clean_mbti(data)

#split in 4 dimensions
EI, NS, TF, JP = dimension_4x.text_split(cleaned)

#text and labels
JP_x = JP['posts']
JP_y = JP['type']

#process raw text into ML compatible features
X = extraction.feature_Tfidf(JP_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, JP_y)

l_svc = LinearSVC(random_state = 0, C = 100, penalty = 'l2')

t0 = time.time()
l_svc.fit(X_train,y_train)
t1 = time.time() # ending time
l_svc_train_time = t1-t0

l_svc_score = l_svc.score(X_test,y_test)

t0 = time.time()
y_true, y_pred_lSVC = y_test, l_svc.predict(X_test)
t1 = time.time() # ending time
l_svc_pred_time = t1-t0

l_svc_report = classification_report(y_true, y_pred_lSVC, output_dict=True)
df_l_svc = pd.DataFrame(l_svc_report)

print(df_l_svc)