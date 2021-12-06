import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from src.data import encode
from src.data import clean_text
from src.data import train_val_test
from src.features import extraction

rand_seed = 42

#import data
data = pd.read_csv(r"C:\Users\jaypi\JayLocalGit\DSE_511_temp\mbti_1.csv")

#Clean Data
cleaned = clean_text.clean_mbti(data)

#Encode classification labels
cleaned = encode.label(cleaned)

#Split into text and labels
X = cleaned['posts']
y = cleaned['type']

#process raw text into ML compatible features
X = extraction.feature_Tfidf(X)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, y)

model = RandomForestClassifier(random_state = 0, n_estimators = 200)
# Number of estimators
n_estimators = [100, 1000]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5] #10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2] #4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
grid = {
               'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)

print("fitting")

grid_result = grid_search.fit(X_val,y_val)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))