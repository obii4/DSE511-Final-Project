import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
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

#text and labels
X = cleaned['posts']
y = cleaned['type']

#process raw text into ML compatible features
X = extraction.feature_Tfidf(X)

#split text data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test.split(X, y)

model = LinearSVC(random_state = 0)
penalty = ['l2','l1']
#loss = ['hinge', 'squared_hinge']
c_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

t = [1e-8]
#define grid search
grid = dict(penalty=penalty, C=c_values, tol=t) #loss=loss,
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_val,y_val)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))