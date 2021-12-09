

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
roc_auc6 = metrics.auc(fpr6, tpr6)
roc_auc7 = metrics.auc(fpr7, tpr7)
roc_auc8 = metrics.auc(fpr8, tpr8)
roc_auc9 = metrics.auc(fpr9, tpr9)
roc_auc10 = metrics.auc(fpr10, tpr10)
roc_auc11 = metrics.auc(fpr11, tpr11)
roc_auc12 = metrics.auc(fpr12, tpr12)
roc_auc13 = metrics.auc(fpr13, tpr13)
roc_auc14 = metrics.auc(fpr14, tpr14)
roc_auc15 = metrics.auc(fpr15, tpr15)

#{'ENFJ': 0, 'ENFP': 1, 'ENTJ': 2, 'ENTP': 3, 'ESFJ': 4, 'ESFP': 5, 'ESTJ': 6, 'ESTP': 7, 'INFJ': 8, 'INFP': 9,
# 'INTJ': 10, 'INTP': 11, 'ISFJ': 12, 'ISFP': 13, 'ISTJ': 14, 'ISTP': 15}

plt.title('XGB ROC')
plt.plot(fpr0, tpr0, 'b--', label = 'ISTP, AUC = %0.2f' % roc_auc0)
plt.plot(fpr1, tpr1, 'b', label = 'ISTJ, AUC = %0.2f' % roc_auc1)
plt.plot(fpr2, tpr2, 'r--', label = 'ISFP, AUC = %0.2f' % roc_auc2)
plt.plot(fpr3, tpr3, 'r', label = 'ISFJ, AUC = %0.2f' % roc_auc3)
plt.plot(fpr4, tpr4, 'g--', label = 'INTP, AUC = %0.2f' % roc_auc4)
plt.plot(fpr5, tpr5, 'g', label = 'INTJ, AUC = %0.2f' % roc_auc5)
plt.plot(fpr6, tpr6, 'c--', label = 'INFP, AUC = %0.2f' % roc_auc6)
plt.plot(fpr7, tpr7, 'm--', label = 'INFJ, AUC = %0.2f' % roc_auc7)
plt.plot(fpr8, tpr8, 'm', label = 'ESTP, AUC = %0.2f' % roc_auc8)
plt.plot(fpr9, tpr9, 'y--', label = 'ESTJ, AUC = %0.2f' % roc_auc9)
plt.plot(fpr10, tpr10, 'y', label = 'ESFP, AUC = %0.2f' % roc_auc10)
plt.plot(fpr11, tpr11, 'm-.', label = 'ESFJ, AUC = %0.2f' % roc_auc11)
plt.plot(fpr12, tpr12, 'b-.', label = 'ENTP, AUC = %0.2f' % roc_auc12)
plt.plot(fpr13, tpr13, 'k-.', label = 'ENTJ, AUC = %0.2f' % roc_auc13)
plt.plot(fpr14, tpr14, 'k', label = 'ENFP, AUC = %0.2f' % roc_auc14)
plt.plot(fpr15, tpr15, 'k', label = 'ENFJ, AUC = %0.2f' % roc_auc15)

plt.legend(loc=(1.01,0))
plt.plot([0, 1], [0, 1],'m--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.rcParams["figure.figsize"] = (7,5)
plt.tight_layout()
plt.savefig('visualizations/roc/xgb_n_roc.png', dpi=700, bbox_inches = "tight")
plt.show()





