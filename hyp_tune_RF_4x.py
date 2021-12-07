import pandas as pd

from src.data import clean_text
from src.data import dimension_4x
from src.features import extraction
from src.models import rf_hyp


rand_seed = 42

#import data, clean, split in 4 dimensions
data = pd.read_csv('~/Desktop/mbti_1.csv')
cleaned = clean_text.clean_mbti(data)
EI, NS, TF, JP = dimension_4x.text_split(cleaned)

## EI Logistic Regression Tuning ##
#EI_para = rf_hyp.tune(EI)

## NS Logistic Regression Tuning ##
NS_para = rf_hyp.tune(NS)

## TF Logistic Regression Tuning ##
TF_para = rf_hyp.tune(TF)

## JP Logistic Regression Tuning ##
JP_para = rf_hyp.tune(JP)
