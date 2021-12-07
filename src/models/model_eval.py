import pandas as pd
import time
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def eval(model, X_train, y_train, X_test, y_test):
    '''Trains and tests the model to produce final
    classification results.'''

    # Train the model using training set while timing
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()  # ending time
    train_time = t1 - t0

    # Test the model while timing
    t2 = time.time()
    y_true, y_pred = y_test, model.predict(X_test)
    t3 = time.time()  # ending time
    test_time = t2 - t3

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report)

    return [train_time, test_time], [y_true, y_pred], report_df