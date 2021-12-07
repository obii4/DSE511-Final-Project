import pandas as pd
from src.data import clean_text
from src.data import encode
from src.features import extraction
from src.models import log_reg_hyp
from src.models import lSVM_hyp
from src.models import XGB_hyp
from src.models import rf_hyp


rand_seed = 42

#import data, clean, encode
data = pd.read_csv('~/Desktop/mbti_1.csv')
cleaned = clean_text.clean_mbti(data)
data_en = encode.label(cleaned)

##Logistic Regression Tuning ##
log_para = log_reg_hyp.tune(data_en)

## Linear SVM Tuning ##
svm_para = lSVM_hyp.tune(data_en)

## Random Forest Tuning ##
rf_para = rf_hyp.tune(data_en)

## XGBoost Tuning ##
xgb_para = XGB_hyp.tune(data_en)
