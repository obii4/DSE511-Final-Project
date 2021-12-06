import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

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
EI_x = EI['posts']
EI_y = EI['type']

#process raw text into ML compatible features
X = extraction.feature_Tfidf(EI_x)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, EI_y)

model = LogisticRegression(random_state=rand_seed, max_iter=1000)
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'saga', 'sag']
penalty = ['l2']
c_values = [65, 66.36363636, 67.72727273, 69.09090909, 70.45454545, 71.81818182,
 73.18181818, 74.54545455, 75.90909091, 77.27272727, 78.63636364, 80]

#define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X_val,y_val)

#summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))