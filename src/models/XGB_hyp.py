import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.data import clean_text
from src.data import dimension_4x
from src.data import train_val_test
from src.features import extraction
from src.data import encode


def tune(Z):
    '''tunes xgboost models'''
    X = Z['posts']
    y = Z['type']

    #process raw text into ML compatible features
    X = extraction.feature_Tfidf(X)

    #split text data
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, y)

    model = XGBClassifier(use_label_encoder=False, random_state=0, eval_metric="merror")
    subsample = [0.5]
    max_depth = [5, 10]
    eta = [0.05, 0.1]

    # define grid search
    grid = dict(max_depth=max_depth, eta=eta, subsample=subsample)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_val, y_val)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result.best_params_