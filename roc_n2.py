

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import sklearn.metrics as metrics



with open("src/models/results/LG_NS_labels.pkl", 'rb') as f:
    lg_labels = pickle.load(f)
    
with open("src/models/results/RF_NS_labels.pkl", 'rb') as f:
    rf_labels = pickle.load(f)

with open("src/models/results/XGB_NS_labels.pkl", 'rb') as f:
    xgb_labels = pickle.load(f)

#find fpr and tpr
prob_lg = lg_labels[2]
fpr_lg_0, tpr_lg_0, threshold = metrics.roc_curve(lg_labels[0], prob_lg[:,0])
fpr_lg_1, tpr_lg_1, threshold = metrics.roc_curve(lg_labels[0], prob_lg[:,1])
prob_rf = rf_labels[2]
fpr_rf_0, tpr_rf_0, threshold = metrics.roc_curve(rf_labels[0], prob_rf[:,0])
fpr_rf_1, tpr_rf_1, threshold = metrics.roc_curve(rf_labels[0], prob_rf[:,1])
prob_xgb = xgb_labels[2]
fpr_xgb_0, tpr_xgb_0, threshold = metrics.roc_curve(xgb_labels[0], prob_xgb[:,0])
fpr_xgb_1, tpr_xgb_1, threshold = metrics.roc_curve(xgb_labels[0], prob_xgb[:,1])

#roc
roc_auc0_lg = metrics.auc(fpr_lg_0, tpr_lg_0)
roc_auc1_lg = metrics.auc(fpr_lg_1, tpr_lg_1)
roc_auc0_rf = metrics.auc(fpr_rf_0, tpr_rf_0)
roc_auc1_rf = metrics.auc(fpr_rf_1, tpr_rf_1)
roc_auc0_xgb = metrics.auc(fpr_xgb_0, tpr_xgb_0)
roc_auc1_xgb = metrics.auc(fpr_xgb_1, tpr_xgb_1)


plt.title('N/s ROC')
plt.plot(fpr_lg_0, tpr_lg_0, 'b--', label = 'LG Class S, AUC = %0.2f' % roc_auc0_lg)
plt.plot(fpr_lg_1, tpr_lg_1, 'b', label = 'LG Class N, AUC = %0.2f' % roc_auc1_lg)

plt.plot(fpr_rf_0, tpr_rf_0, 'r--', label = 'RF Class S, AUC = %0.2f' % roc_auc0_rf)
plt.plot(fpr_rf_1, tpr_rf_1, 'r', label = 'RF Class N, AUC = %0.2f' % roc_auc1_rf)

plt.plot(fpr_xgb_0, tpr_xgb_0, 'g--', label = 'XGB Class S, AUC = %0.2f' % roc_auc0_xgb)
plt.plot(fpr_xgb_1, tpr_xgb_1, 'g', label = 'XGB Class N, AUC = %0.2f' % roc_auc1_xgb)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'m--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.rcParams["figure.figsize"] = (10,5)
plt.savefig('visualizations/roc/ns_roc.png', dpi=700)
plt.show()





