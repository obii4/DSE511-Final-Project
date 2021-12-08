

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import sklearn.metrics as metrics




with open("src/models/results/xgb_16_labels.pkl", 'rb') as f:
    lg_labels = pickle.load(f)


prob_lg = lg_labels[2]

fpr0, tpr0, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 0], pos_label=15)
fpr1, tpr1, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 1], pos_label=14)
fpr2, tpr2, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 2], pos_label=13)
fpr3, tpr3, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 3], pos_label=12)
fpr4, tpr4, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 4], pos_label=11)
fpr5, tpr5, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 5], pos_label=10)
fpr6, tpr6, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 6], pos_label=9)
fpr7, tpr7, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 7], pos_label=8)
fpr8, tpr8, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 8], pos_label=7)
fpr9, tpr9, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 9], pos_label=6)
fpr10, tpr10, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 10], pos_label=5)
fpr11, tpr11, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 11], pos_label=4)
fpr12, tpr12, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 12], pos_label=3)
fpr13, tpr13, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 13], pos_label=2)
fpr14, tpr14, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 14], pos_label=1)
fpr15, tpr15, _ = metrics.roc_curve(lg_labels[0], prob_lg[:, 15], pos_label=0)


roc_auc0 = metrics.auc(fpr0, tpr0)
roc_auc1 = metrics.auc(fpr1, tpr1)
roc_auc2 = metrics.auc(fpr2, tpr2)
roc_auc3 = metrics.auc(fpr3, tpr3)
roc_auc4 = metrics.auc(fpr4, tpr4)
roc_auc5 = metrics.auc(fpr5, tpr5)


#{'ENFJ': 0, 'ENFP': 1, 'ENTJ': 2, 'ENTP': 3, 'ESFJ': 4, 'ESFP': 5, 'ESTJ': 6, 'ESTP': 7, 'INFJ': 8, 'INFP': 9,
# 'INTJ': 10, 'INTP': 11, 'ISFJ': 12, 'ISFP': 13, 'ISTJ': 14, 'ISTP': 15}

plt.title('XGBoost ROC')
plt.plot(fpr0, tpr0, 'b--', label = 'ISTP')
plt.plot(fpr1, tpr1, 'b', label = 'ISTJ')
plt.plot(fpr2, tpr2, 'r--', label = 'ISFP')
plt.plot(fpr3, tpr3, 'r', label = 'ISFJ')
plt.plot(fpr4, tpr4, 'g--', label = 'INTP')
plt.plot(fpr5, tpr5, 'g', label = 'INTJ')
plt.plot(fpr6, tpr6, 'c--', label = 'INFP')
plt.plot(fpr7, tpr7, 'm--', label = 'INFJ')
plt.plot(fpr8, tpr8, 'm', label = 'ESTP')
plt.plot(fpr9, tpr9, 'y--', label = 'ESTJ')
plt.plot(fpr10, tpr10, 'y', label = 'ESFP')
plt.plot(fpr11, tpr11, 'm-.', label = 'ESFJ')
plt.plot(fpr12, tpr12, 'b-.', label = 'ENTP')
plt.plot(fpr13, tpr13, 'k-.', label = 'ENTJ')
plt.plot(fpr14, tpr14, 'k', label = 'ENFP')
plt.plot(fpr15, tpr15, 'k', label = 'ENFJ')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'m--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.rcParams["figure.figsize"] = (10,5)
plt.savefig('visualizations/roc/xgb_roc.png', dpi=700)
plt.show()





