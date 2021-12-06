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

#import data
data = pd.read_csv('~/Desktop/mbti_1.csv')
cleaned = clean_text.clean_mbti(data)













###
###
###
#split in 4 dimensions
EI, NS, TF, JP = dimension_4x.text_split(cleaned)

#MBTI Dimensions text and labels
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
X = extraction.feature_Tfidf(EI_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, EI_y)

lg = LogisticRegression(random_state=0, C=73.18181818, penalty='l2', solver = 'saga', max_iter=1000)

t0 = time.time()
lg.fit(X_train,y_train)
t1 = time.time() # ending time
lg_train_time = t1-t0

t0 = time.time()
y_true, y_pred_lg = y_test, lg.predict(X_test)
t1 = time.time() # ending time
lg_pred_time = t1-t0

lg_report = classification_report(y_true, y_pred_lg, output_dict=True)
df_lg_ei = pd.DataFrame(lg_report)
