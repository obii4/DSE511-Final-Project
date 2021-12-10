# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 02:56:08 2021

@author: Student User
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data import train_val_test
from src.data import clean_text
from collections import Counter
from itertools import chain
import collections
#import data
data = pd.read_csv('~/Desktop/mbti_1.csv')
cleaned = clean_text.clean_mbti(data)
label_encoder = LabelEncoder()
cleaned['type'] = label_encoder.fit_transform(cleaned['type'])

data_en = cleaned
#data_en = encode.label(cleaned) #labels encoded

#text and labels
all_x = data_en['posts']
all_y = data_en['type']

#process raw text into ML compatible features
vectorizer = TfidfVectorizer(min_df=3, stop_words='english',
                                 ngram_range=(1, 2), lowercase=True,max_df=0.1)
vectorizer.fit(all_x)

vec = vectorizer.transform(all_x)

#Most common words per dimension
print('Common words per personality type')
#INTJ
realtype = label_encoder.inverse_transform(data_en['type'])
INTJ = np.where(realtype == 'INTJ')
words = vectorizer.inverse_transform(vec[INTJ])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
INTJ_words = R
#INTP
INTP = np.where(realtype == 'INTP')
words = vectorizer.inverse_transform(vec[INTP])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
INTP_words = R
#INFJ
INFJ = np.where(realtype == 'INFJ')
words = vectorizer.inverse_transform(vec[INFJ])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
INFJ_words = R
#INFP
INFP = np.where(realtype == 'INFP')
words = vectorizer.inverse_transform(vec[INFP])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
INFP_words = R
#ISTJ
ISTJ = np.where(realtype == 'ISTJ')
words = vectorizer.inverse_transform(vec[ISTJ])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ISTJ_words = R
#ISTP
ISTP = np.where(realtype == 'ISTP')
words = vectorizer.inverse_transform(vec[ISTP])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ISTP_words = R
#ISFJ
ISFJ = np.where(realtype == 'ISFJ')
words = vectorizer.inverse_transform(vec[ISFJ])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ISFJ_words = R
#ISFP
ISFP = np.where(realtype == 'ISFP')
words = vectorizer.inverse_transform(vec[ISFP])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ISFP_words = R
#ENTJ
ENTJ = np.where(realtype == 'ENTJ')
words = vectorizer.inverse_transform(vec[ENTJ])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ENTJ_words = R
#ENTP
ENTP = np.where(realtype == 'ENTP')
words = vectorizer.inverse_transform(vec[ENTP])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ENTP_words = R
#EINFJ
ENFJ = np.where(realtype == 'ENFJ')
words = vectorizer.inverse_transform(vec[ENFJ])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ENFJ_words = R
#ENFP
ENFP = np.where(realtype == 'ENFP')
words = vectorizer.inverse_transform(vec[ENFP])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ENFP_words = R
#ESTJ
ESTJ = np.where(realtype == 'ESTJ')
words = vectorizer.inverse_transform(vec[ESTJ])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ESTJ_words = R
#ESTP
ESTP = np.where(realtype == 'ESTP')
words = vectorizer.inverse_transform(vec[ESTP])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ESTP_words = R
#ESFJ
ESFJ = np.where(realtype == 'ESFJ')
words = vectorizer.inverse_transform(vec[ESFJ])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ESFJ_words = R
#ESFP
ESFP = np.where(realtype == 'ESFP')
words = vectorizer.inverse_transform(vec[ESFP])
Q = Counter(chain.from_iterable(words))
A = Q.most_common(30)
zip(A)
R = np.asarray(A)
ESFP_words = R


#Combine top 15 for each type
top = (INTJ_words[:,0], INTP_words[:,0], INFP_words[:,0], INFJ_words[:,0], 
                     ISTJ_words[:,0], ISTP_words[:,0], ISFP_words[:,0], ISFJ_words[:,0], 
                     ENTJ_words[:,0], ENTP_words[:,0], ENFP_words[:,0], ENFJ_words[:,0],
                     ESTJ_words[:,0], ESTP_words[:,0], ESFP_words[:,0], ESFJ_words[:,0])
top = np.transpose(top)
top_df = pd.DataFrame(data = top, columns= ['INTJ','INTP','INFP','INFJ',
                                          'ISTJ','ISTP','ISFP','ISFJ',
                                          'ENTJ','ENTP','ENFP','ENFJ',
                                          'ESTJ','ESTP','ESFP','ESFJ'])
print(top_df)