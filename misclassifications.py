# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:23:16 2021

@author: Student User
"""
#Generate Misclassifications Data
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
                                 ngram_range=(1, 2), lowercase=True)
vectorizer.fit(all_x)

vec = vectorizer.transform(all_x)


#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(vec, all_y)

### Logistic Regression ###
LG = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
LG.fit(X_train,y_train)
y_true, y_pred = y_test, LG.predict(X_test)
y_true = np.asarray(y_true)
misclass=np.where(y_true != LG.predict(X_test))
true_label = label_encoder.inverse_transform(y_true)
guess = label_encoder.inverse_transform(y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Types
a = label_encoder.inverse_transform(y_true[misclass])
#print(a)
B = Counter(a)
B.most_common()
print('Misclassified type for Logistic Regression')
print(B)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(X_test[misclass])
#print(wrongwords)
import collections, numpy
print('Misclassified words for Logisitc Regression')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(100)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
print(R)
###Linear SVM###
LG = LinearSVC(random_state=0, C=10, penalty='l2')
LG.fit(X_train,y_train)
y_true, y_pred = y_test, LG.predict(X_test)
y_true = np.asarray(y_true)
misclass=np.where(y_true != LG.predict(X_test))
true_label = label_encoder.inverse_transform(y_true)
guess = label_encoder.inverse_transform(y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Types
a = label_encoder.inverse_transform(y_true[misclass])
#print(a)
B = Counter(a)
B.most_common()
print('Misclassified type for Linear SVM')
print(B)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(X_test[misclass])
#print(wrongwords)
import collections, numpy
print('Misclassified words for Linear SVM')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(100)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
print(R)

###Random Forest###
LG = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=None, min_samples_leaf=2, min_samples_split=5, n_estimators=100)
LG.fit(X_train,y_train)
y_true, y_pred = y_test, LG.predict(X_test)
y_true = np.asarray(y_true)
misclass=np.where(y_true != LG.predict(X_test))
true_label = label_encoder.inverse_transform(y_true)
guess = label_encoder.inverse_transform(y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Types
a = label_encoder.inverse_transform(y_true[misclass])
#print(a)
B = Counter(a)
B.most_common()
print('Misclassified type for Randome Forest')
print(B)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(X_test[misclass])
#print(wrongwords)
import collections, numpy
print('Misclassified words for Random Forest')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(100)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
print(R)




### XGBoost ###
LG = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.05, max_depth=5, subsample=.5)
LG.fit(X_train,y_train)
y_true, y_pred = y_test, LG.predict(X_test)
y_true = np.asarray(y_true)
misclass=np.where(y_true != LG.predict(X_test))
true_label = label_encoder.inverse_transform(y_true)
guess = label_encoder.inverse_transform(y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Types
a = label_encoder.inverse_transform(y_true[misclass])
#print(a)
B = Counter(a)
B.most_common()
print('Misclassified type for XGBoost')
print(B)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(X_test[misclass])
#print(wrongwords)
import collections, numpy
print('Misclassified words for XGBoost')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(100)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
print(R)



 
