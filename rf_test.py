import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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
NS_x = NS['posts']
NS_y = NS['type']

#process raw text into ML compatible features
X = extraction.feature_Tfidf(NS_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, NS_y)

rf = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2)

t0 = time.time()
rf.fit(X_train,y_train)
t1 = time.time() # ending time
rf_train_time = t1-t0

t0 = time.time()
y_true, y_pred_rf = y_test, rf.predict(X_test)
t1 = time.time() # ending time
rf_pred_time = t1-t0

rf_report = classification_report(y_true, y_pred_rf, output_dict=True)
df_rf = pd.DataFrame(rf_report)

print(df_rf)