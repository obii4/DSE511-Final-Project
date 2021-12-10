# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from IPython.display import display
from sklearn.preprocessing import LabelEncoder
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
#Split data
text=cleaned
extroversion = text[text['type'].isin(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP'])]
extroversion = extroversion.replace(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP'], 
                                    [0, 0, 0, 0, 0, 0, 0, 0])
introversion = text[text['type'].isin(['INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'])]
introversion = introversion.replace(['INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'], 
                                    [1, 1, 1, 1, 1, 1, 1, 1])

intuition = text[text['type'].isin(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'INTJ', 'INTP'])]
intuition = intuition.replace(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'INTJ', 'INTP'], 
                                    [0, 0, 0, 0, 0, 0, 0, 0])
sensing = text[text['type'].isin(['ISFJ', 'ISFP', 'ISTJ', 'ISTP','ESFJ', 'ESFP', 'ESTJ', 'ESTP'])]
sensing = sensing.replace(['ISFJ', 'ISFP', 'ISTJ', 'ISTP','ESFJ', 'ESFP', 'ESTJ', 'ESTP'], 
                                    [1, 1, 1, 1, 1, 1, 1, 1])

thinking = text[text['type'].isin(['ENTJ', 'ENTP','ESTJ', 'ESTP','INTJ', 'INTP', 'ISTJ', 'ISTP'])]
thinking = thinking.replace(['ENTJ', 'ENTP','ESTJ', 'ESTP','INTJ', 'INTP', 'ISTJ', 'ISTP'], 
                                    [0, 0, 0, 0, 0, 0, 0, 0])
feeling = text[text['type'].isin(['ENFJ', 'ENFP','ESFJ', 'ESFP', 'INFJ', 'INFP', 'ISFJ', 'ISFP'])]
feeling = feeling.replace(['ENFJ', 'ENFP','ESFJ', 'ESFP', 'INFJ', 'INFP', 'ISFJ', 'ISFP'], 
                                    [1, 1, 1, 1, 1, 1, 1, 1])

judging = text[text['type'].isin(['ENFJ','ENTJ', 'ESFJ', 'ESTJ', 'INFJ', 'INTJ', 'ISFJ', 'ISTJ'])]
judging = judging.replace(['ENFJ','ENTJ', 'ESFJ', 'ESTJ', 'INFJ', 'INTJ', 'ISFJ', 'ISTJ'], 
                                    [0, 0, 0, 0, 0, 0, 0, 0])

percieving = text[text['type'].isin(['ENFP', 'ENTP', 'ESFP', 'ESTP', 'INFP', 'INTP', 'ISFP', 'ISTP'])]
percieving = percieving.replace(['ENFP', 'ENTP', 'ESFP', 'ESTP', 'INFP', 'INTP', 'ISFP', 'ISTP'], 
                                    [1, 1, 1, 1, 1, 1, 1, 1])
    
EI = pd.concat([extroversion, introversion])
NS = pd.concat([intuition, sensing])
TF = pd.concat([thinking, feeling])
JP = pd.concat([judging, percieving])


##MBTI Dimensions text and labels
#Extraversion (E) / Introversion (I)
EI['type'] = label_encoder.fit_transform(EI['type'])
EI_x = EI['posts']
EI_y = EI['type']
#Intuition (N) / Sensing (S)
NS['type'] = label_encoder.fit_transform(NS['type'])
NS_x = NS['posts']
NS_y = NS['type']
#Thinking (T) / Feeling (F)
TF['type'] = label_encoder.fit_transform(TF['type'])
TF_x = TF['posts']
TF_y = TF['type']
#Judging (J) / Perceiving (P)
JP['type'] = label_encoder.fit_transform(JP['type'])
JP_x = JP['posts']
JP_y = JP['type']


#process raw text into ML compatible features
vectorizer = TfidfVectorizer(min_df=3, stop_words='english',
                                 ngram_range=(1, 2), lowercase=True)
vectorizer.fit(EI_x)
X_EI = vectorizer.transform(EI_x)
vectorizer.fit(NS_x)
X_NS = vectorizer.transform(NS_x)
vectorizer.fit(TF_x)
X_TF = vectorizer.transform(TF_x)
vectorizer.fit(JP_x)
X_JP = vectorizer.transform(JP_x)



#split text data
EI_X_train, __, EI_X_test, EI_y_train, __, EI_y_test = train_val_test.split(X_EI, EI_y)
NS_X_train, __, NS_X_test, NS_y_train, __, NS_y_test = train_val_test.split(X_NS, NS_y)
TF_X_train, __, TF_X_test, TF_y_train, __, TF_y_test = train_val_test.split(X_TF, TF_y)
JP_X_train, __, JP_X_test, JP_y_train, __, JP_y_test = train_val_test.split(X_JP, JP_y)


### EI ###
print('Misclassified words for Extroverted vs Introverted')

LG = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
LSVC = LinearSVC(random_state = 0, C = 0.0001, penalty = 'l2')
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.1, max_depth=10, subsample=1)
RF = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=10, min_samples_leaf=1, min_samples_split=2)

### LogisticRegression ###
LG.fit(EI_X_train,EI_y_train)
EI_y_true, EI_y_pred = EI_y_test, LG.predict(EI_X_test)
EI_y_true = np.asarray(EI_y_true)
misclass=np.where(EI_y_true != LG.predict(EI_X_test))
true_label = label_encoder.inverse_transform(EI_y_true)
guess = label_encoder.inverse_transform(EI_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(EI_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for Logisitc Regression')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
EI_LG = R

### LinearSVM ###
LSVC.fit(EI_X_train,EI_y_train)
EI_y_true, EI_y_pred = EI_y_test, LSVC.predict(EI_X_test)
EI_y_true = np.asarray(EI_y_true)
misclass=np.where(EI_y_true != LSVC.predict(EI_X_test))
true_label = label_encoder.inverse_transform(EI_y_true)
guess = label_encoder.inverse_transform(EI_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(EI_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for LinearSVM')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
EI_SVM = R

### XGBoost ###
XGB.fit(EI_X_train,EI_y_train)
EI_y_true, EI_y_pred = EI_y_test, XGB.predict(EI_X_test)
EI_y_true = np.asarray(EI_y_true)
misclass=np.where(EI_y_true != XGB.predict(EI_X_test))
true_label = label_encoder.inverse_transform(EI_y_true)
guess = label_encoder.inverse_transform(EI_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(EI_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for XGBoost')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
EI_XGB = R

### Random Forest ###
RF.fit(EI_X_train,EI_y_train)
EI_y_true, EI_y_pred = EI_y_test, RF.predict(EI_X_test)
EI_y_true = np.asarray(EI_y_true)
misclass=np.where(EI_y_true != RF.predict(EI_X_test))
true_label = label_encoder.inverse_transform(EI_y_true)
guess = label_encoder.inverse_transform(EI_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(EI_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for Random Forest')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
EI_RF = R

#Combine top 15 missed words for dimension
EI = np.concatenate((EI_LG, EI_SVM, EI_XGB, EI_RF),axis=1)
EI_df = pd.DataFrame(data = EI, columns= ['Logistic Regression:','','SVM:','','XGBoost:','','Random Forest:',''])
print(EI_df)



### NS ###
print('Misclassified words for Intuition versus Sensing')
LG = LogisticRegression(random_state=0, C=10, penalty='l1', solver = 'liblinear', max_iter=1000)
LSVC = LinearSVC(random_state = 0, C = 100, penalty = 'l2')
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.1, max_depth=10, subsample=1)
RF = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100)

### LogisticRegression ###
LG.fit(NS_X_train,NS_y_train)
NS_y_true, NS_y_pred = NS_y_test, LG.predict(NS_X_test)
NS_y_true = np.asarray(NS_y_true)
misclass=np.where(NS_y_true != LG.predict(NS_X_test))
true_label = label_encoder.inverse_transform(NS_y_true)
guess = label_encoder.inverse_transform(NS_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(NS_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for Logisitc Regression')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
NS_LG = R

### LinearSVM ###
LSVC.fit(NS_X_train,NS_y_train)
NS_y_true, NS_y_pred = NS_y_test, LSVC.predict(NS_X_test)
NS_y_true = np.asarray(NS_y_true)
misclass=np.where(NS_y_true != LSVC.predict(NS_X_test))
true_label = label_encoder.inverse_transform(NS_y_true)
guess = label_encoder.inverse_transform(NS_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(NS_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for LinearSVM')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
NS_SVM = R

### XGBoost ###
XGB.fit(NS_X_train,NS_y_train)
NS_y_true, NS_y_pred = NS_y_test, XGB.predict(NS_X_test)
NS_y_true = np.asarray(NS_y_true)
misclass=np.where(NS_y_true != XGB.predict(NS_X_test))
true_label = label_encoder.inverse_transform(NS_y_true)
guess = label_encoder.inverse_transform(NS_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(NS_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for XGBoost')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
NS_XGB = R

### Random Forest ###
RF.fit(NS_X_train,NS_y_train)
NS_y_true, NS_y_pred = NS_y_test, RF.predict(NS_X_test)
NS_y_true = np.asarray(NS_y_true)
misclass=np.where(NS_y_true != RF.predict(NS_X_test))
true_label = label_encoder.inverse_transform(NS_y_true)
guess = label_encoder.inverse_transform(NS_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(NS_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for Random Forest')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
NS_RF = R

#Combine top 15 missed for dimesnion
NS = np.concatenate((NS_LG, NS_SVM, NS_XGB, NS_RF),axis=1)
NS_df = pd.DataFrame(data = NS, columns= ['Logistic Regression:','','SVM:','','XGBoost:','','Random Forest:',''])
print(NS_df)



### TF ###
print('Misclassified terms for Thinking versus Feeling')
LG = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
LSVC = LinearSVC(random_state = 0, C = 10, penalty = 'l2')
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.1, max_depth=10, subsample=.5)
RF = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=60, min_samples_leaf=1, min_samples_split=5, n_estimators=1000)
### LogisticRegression ###
LG.fit(TF_X_train,TF_y_train)
TF_y_true, TF_y_pred = TF_y_test, LG.predict(TF_X_test)
TF_y_true = np.asarray(TF_y_true)
misclass=np.where(TF_y_true != LG.predict(TF_X_test))
true_label = label_encoder.inverse_transform(TF_y_true)
guess = label_encoder.inverse_transform(TF_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(TF_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for Logisitc Regression')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
TF_LG = R

### LinearSVM ###
LSVC.fit(TF_X_train,TF_y_train)
TF_y_true, TF_y_pred = TF_y_test, LSVC.predict(TF_X_test)
TF_y_true = np.asarray(TF_y_true)
misclass=np.where(TF_y_true != LSVC.predict(TF_X_test))
true_label = label_encoder.inverse_transform(TF_y_true)
guess = label_encoder.inverse_transform(TF_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(TF_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for LinearSVM')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
TF_SVM = R

### XGBoost ###
XGB.fit(TF_X_train,TF_y_train)
TF_y_true, TF_y_pred = TF_y_test, XGB.predict(TF_X_test)
TF_y_true = np.asarray(TF_y_true)
misclass=np.where(TF_y_true != XGB.predict(TF_X_test))
true_label = label_encoder.inverse_transform(TF_y_true)
guess = label_encoder.inverse_transform(TF_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(TF_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for XGBoost')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
TF_XGB = R

### Random Forest ###
RF.fit(TF_X_train,TF_y_train)
TF_y_true, TF_y_pred = TF_y_test, RF.predict(TF_X_test)
TF_y_true = np.asarray(TF_y_true)
misclass=np.where(TF_y_true != RF.predict(TF_X_test))
true_label = label_encoder.inverse_transform(TF_y_true)
guess = label_encoder.inverse_transform(TF_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(TF_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for Random Forest')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
TF_RF = R

#Combine top 15 missed for dimesnion
TF = np.concatenate((TF_LG, TF_SVM, TF_XGB, TF_RF),axis=1)
TF_df = pd.DataFrame(data = TF, columns= ['Logistic Regression:','','SVM:','','XGBoost:','','Random Forest:',''])
print(TF_df)



### JP ###
print('Misclassified terms for judging versus perceiving')
LG = LogisticRegression(random_state=0, C=100, penalty='l2', solver = 'liblinear', max_iter=1000)
LSVC = LinearSVC(random_state = 0, C = 10, penalty = 'l2')
XGB = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror", eta=0.05, max_depth=10, subsample=1)
RF = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=60, min_samples_leaf=1, min_samples_split=5, n_estimators=100)

### LogisticRegression ###
LG.fit(JP_X_train,JP_y_train)
JP_y_true, JP_y_pred = JP_y_test, LG.predict(JP_X_test)
JP_y_true = np.asarray(JP_y_true)
misclass=np.where(JP_y_true != LG.predict(JP_X_test))
true_label = label_encoder.inverse_transform(JP_y_true)
guess = label_encoder.inverse_transform(JP_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(JP_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for Logisitc Regression')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
JP_LG = R

### LinearSVM ###
LSVC.fit(JP_X_train,JP_y_train)
JP_y_true, JP_y_pred = JP_y_test, LSVC.predict(JP_X_test)
JP_y_true = np.asarray(JP_y_true)
misclass=np.where(JP_y_true != LSVC.predict(JP_X_test))
true_label = label_encoder.inverse_transform(JP_y_true)
guess = label_encoder.inverse_transform(JP_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(JP_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for LinearSVM')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
JP_SVM = R

### XGBoost ###
XGB.fit(JP_X_train,JP_y_train)
JP_y_true, JP_y_pred = JP_y_test, XGB.predict(JP_X_test)
JP_y_true = np.asarray(JP_y_true)
misclass=np.where(JP_y_true != XGB.predict(JP_X_test))
true_label = label_encoder.inverse_transform(JP_y_true)
guess = label_encoder.inverse_transform(JP_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(JP_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for XGBoost')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
JP_XGB = R

### Random Forest ###
RF.fit(JP_X_train,JP_y_train)
JP_y_true, JP_y_pred = JP_y_test, RF.predict(JP_X_test)
JP_y_true = np.asarray(JP_y_true)
misclass=np.where(JP_y_true != RF.predict(JP_X_test))
true_label = label_encoder.inverse_transform(JP_y_true)
guess = label_encoder.inverse_transform(JP_y_pred)
misclass=np.where(true_label != guess)
#print(misclass)

#Misclassified Words
wrongwords = vectorizer.inverse_transform(JP_X_test[misclass])
#print(wrongwords)
import collections, numpy
#print('Misclassified words for Random Forest')
Q = Counter(chain.from_iterable(wrongwords))
A = Q.most_common(15)
#print(A)
zip(A)
#print(A)
R = np.asarray(A)
#print(R)
JP_RF = R

#Combine top 15 missed for dimesnion
JP = np.concatenate((JP_LG, JP_SVM, JP_XGB, JP_RF),axis=1)
JP_df = pd.DataFrame(data = JP, columns= ['Logistic Regression:','','SVM:','','XGBoost:','','Random Forest:',''])
print(JP_df)



 
# Generate Confusion Matrix 
# import seaborn as sn
# cm1 = confusion_matrix(y_test,y_pred)
# fig, ax = plt.subplots(figsize=(16,16))
# sn.heatmap(cm1, annot=True, fmt='d',
#            xticklabels=cleaned.type,
#            yticklabels=cleaned.type)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()
# disp = ConfusionMatrixDisplay(cm1, display_labels = type)
# disp.plot()
