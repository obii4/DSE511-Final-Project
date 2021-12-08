import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

### ALL ###
### Classification Report ###
with open("src/models/results/LG_16_class_results.pkl", 'rb') as f:
    LG_16_results = pickle.load(f)
    LG_16_results = LG_16_results.round(4)

with open("src/models/results/LSVC_16_class_results.pkl", 'rb') as f:
    LSVM_16_results = pickle.load(f)
    LSVM_16_results = LSVM_16_results.round(4)

with open("src/models/results/RF_16_class_results.pkl", 'rb') as f:
    RF_16_results = pickle.load(f)
    RF_16_results = RF_16_results.round(4)

with open("src/models/results/XGB_16_class_results.pkl", 'rb') as f:
    XGB_16_results = pickle.load(f)
    XGB_16_results = XGB_16_results.round(4)

### train / test times ###
with open("src/models/results/LG_16_times.pkl", 'rb') as f:
    LG_16_times = pickle.load(f)
    LG_16_times = LG_16_times.round(4)

with open("src/models/results/LSVC_16_times.pkl", 'rb') as f:
    LSVM_16_times = pickle.load(f)
    LSVM_16_times = LSVM_16_times.round(4)

with open("src/models/results/RF_16_times.pkl", 'rb') as f:
    RF_16_times = pickle.load(f)
    RF_16_times = RF_16_times.round(4)

with open("src/models/results/XGB_16_times.pkl", 'rb') as f:
    XGB_16_times = pickle.load(f)
    XGB_16_times = XGB_16_times.round(4)

accuracy_16 = pd.DataFrame({'Accuracy': [LG_16_results['accuracy'][1], LSVM_16_results['accuracy'][1], RF_16_results['accuracy'][1], XGB_16_results['accuracy'][1]],
                            'Train Time': [LG_16_times['train_time'][0], LSVM_16_times['train_time'][0], RF_16_times['train_time'][0], XGB_16_times['train_time'][0]],
                            'Test Time': [LG_16_times['test_time'][0], LSVM_16_times['test_time'][0], RF_16_times['test_time'][0], XGB_16_times['test_time'][0]]})
accuracy_16.index = ['Logistic Regression', 'Linear SVM', 'Random Forest', 'XGBoost']
print(accuracy_16)

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)

table(ax, accuracy_16)

plt.savefig('visualizations/tables/summary_16.png',dpi=700, transparent = True, bbox_inches = 'tight', pad_inches = 0)

### E/I ###
### Classification Report ###
with open("src/models/results/LG_EI_class_results.pkl", 'rb') as f:
    LG_EI_results = pickle.load(f)
    LG_EI_results = LG_EI_results.round(4)

with open("src/models/results/LSVC_EI_class_results.pkl", 'rb') as f:
    LSVM_EI_results = pickle.load(f)
    LSVM_EI_results = LSVM_EI_results.round(4)

with open("src/models/results/RF_EI_class_results.pkl", 'rb') as f:
    RF_EI_results = pickle.load(f)
    RF_EI_results = RF_EI_results.round(4)

with open("src/models/results/XGB_EI_class_results.pkl", 'rb') as f:
    XGB_EI_results = pickle.load(f)
    XGB_EI_results = XGB_EI_results.round(4)

### train / test times ###
with open("src/models/results/LG_EI_times.pkl", 'rb') as f:
    LG_EI_times = pickle.load(f)
    LG_EI_times = LG_EI_times.round(4)

with open("src/models/results/LSVC_EI_times.pkl", 'rb') as f:
    LSVM_EI_times = pickle.load(f)
    LSVM_EI_times = LSVM_EI_times.round(4)

with open("src/models/results/RF_EI_times.pkl", 'rb') as f:
    RF_EI_times = pickle.load(f)
    RF_EI_times = RF_EI_times.round(4)

with open("src/models/results/XGB_EI_times.pkl", 'rb') as f:
    XGB_EI_times = pickle.load(f)
    XGB_EI_times = XGB_EI_times.round(4)

accuracy_EI = pd.DataFrame({'Accuracy': [LG_EI_results['accuracy'][1], LSVM_EI_results['accuracy'][1], RF_EI_results['accuracy'][1], XGB_EI_results['accuracy'][1]],
                            'Train Time': [LG_EI_times['train_time'][0], LSVM_EI_times['train_time'][0], RF_EI_times['train_time'][0], XGB_EI_times['train_time'][0]],
                            'Test Time': [LG_EI_times['test_time'][0], LSVM_EI_times['test_time'][0], RF_EI_times['test_time'][0], XGB_EI_times['test_time'][0]]})
accuracy_EI.index = ['Logistic Regression', 'Linear SVM', 'Random Forest', 'XGBoost']
print(accuracy_EI)

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)

table(ax, accuracy_EI)

plt.savefig('visualizations/tables/summary_EI.png',dpi=700, transparent = True, bbox_inches = 'tight', pad_inches = 0)

### T/F ###
### Classification Report ###
with open("src/models/results/LG_TF_class_results.pkl", 'rb') as f:
    LG_TF_results = pickle.load(f)
    LG_TF_results = LG_TF_results.round(4)

with open("src/models/results/LSVC_TF_class_results.pkl", 'rb') as f:
    LSVM_TF_results = pickle.load(f)
    LSVM_TF_results = LSVM_TF_results.round(4)

with open("src/models/results/RF_TF_class_results.pkl", 'rb') as f:
    RF_TF_results = pickle.load(f)
    RF_TF_results = RF_TF_results.round(4)

with open("src/models/results/XGB_TF_class_results.pkl", 'rb') as f:
    XGB_TF_results = pickle.load(f)
    XGB_TF_results = XGB_TF_results.round(4)

### train / test times ###
with open("src/models/results/LG_TF_times.pkl", 'rb') as f:
    LG_TF_times = pickle.load(f)
    LG_TF_times = LG_TF_times.round(4)

with open("src/models/results/LSVC_TF_times.pkl", 'rb') as f:
    LSVM_TF_times = pickle.load(f)
    LSVM_TF_times = LSVM_TF_times.round(4)

with open("src/models/results/RF_TF_times.pkl", 'rb') as f:
    RF_TF_times = pickle.load(f)
    RF_TF_times = RF_TF_times.round(4)

with open("src/models/results/XGB_TF_times.pkl", 'rb') as f:
    XGB_TF_times = pickle.load(f)
    XGB_TF_times = XGB_TF_times.round(4)

accuracy_TF = pd.DataFrame({'Accuracy': [LG_TF_results['accuracy'][1], LSVM_TF_results['accuracy'][1], RF_TF_results['accuracy'][1], XGB_TF_results['accuracy'][1]],
                            'Train Time': [LG_TF_times['train_time'][0], LSVM_TF_times['train_time'][0], RF_TF_times['train_time'][0], XGB_TF_times['train_time'][0]],
                            'Test Time': [LG_TF_times['test_time'][0], LSVM_TF_times['test_time'][0], RF_TF_times['test_time'][0], XGB_TF_times['test_time'][0]]})
accuracy_TF.index = ['Logistic Regression', 'Linear SVM', 'Random Forest', 'XGBoost']
print(accuracy_TF)

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)

table(ax, accuracy_TF)

plt.savefig('visualizations/tables/summary_TF.png',dpi=700, transparent = True, bbox_inches = 'tight', pad_inches = 0)

### N/S ###
### Classification Report ###
with open("src/models/results/LG_NS_class_results.pkl", 'rb') as f:
    LG_NS_results = pickle.load(f)
    LG_NS_results = LG_NS_results.round(4)

with open("src/models/results/LSVC_NS_class_results.pkl", 'rb') as f:
    LSVM_NS_results = pickle.load(f)
    LSVM_NS_results = LSVM_NS_results.round(4)

with open("src/models/results/RF_NS_class_results.pkl", 'rb') as f:
    RF_NS_results = pickle.load(f)
    RF_NS_results = RF_NS_results.round(4)

with open("src/models/results/XGB_NS_class_results.pkl", 'rb') as f:
    XGB_NS_results = pickle.load(f)
    XGB_NS_results = XGB_NS_results.round(4)

### train / test times ###
with open("src/models/results/LG_NS_times.pkl", 'rb') as f:
    LG_NS_times = pickle.load(f)
    LG_NS_times = LG_NS_times.round(4)

with open("src/models/results/LSVC_NS_times.pkl", 'rb') as f:
    LSVM_NS_times = pickle.load(f)
    LSVM_NS_times = LSVM_NS_times.round(4)

with open("src/models/results/RF_NS_times.pkl", 'rb') as f:
    RF_NS_times = pickle.load(f)
    RF_NS_times = RF_NS_times.round(4)

with open("src/models/results/XGB_NS_times.pkl", 'rb') as f:
    XGB_NS_times = pickle.load(f)
    XGB_NS_times = XGB_NS_times.round(4)

accuracy_NS = pd.DataFrame({'Accuracy': [LG_NS_results['accuracy'][1], LSVM_NS_results['accuracy'][1], RF_NS_results['accuracy'][1], XGB_NS_results['accuracy'][1]],
                            'Train Time': [LG_NS_times['train_time'][0], LSVM_NS_times['train_time'][0], RF_NS_times['train_time'][0], XGB_NS_times['train_time'][0]],
                            'Test Time': [LG_NS_times['test_time'][0], LSVM_NS_times['test_time'][0], RF_NS_times['test_time'][0], XGB_NS_times['test_time'][0]]})
accuracy_NS.index = ['Logistic Regression', 'Linear SVM', 'Random Forest', 'XGBoost']
print(accuracy_NS)

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)

table(ax, accuracy_NS)

plt.savefig('visualizations/tables/summary_NS.png',dpi=700, transparent = True, bbox_inches = 'tight', pad_inches = 0)

### J/P ###
### Classification Report ###
with open("src/models/results/LG_JP_class_results.pkl", 'rb') as f:
    LG_JP_results = pickle.load(f)
    LG_JP_results = LG_JP_results.round(4)

with open("src/models/results/LSVC_JP_class_results.pkl", 'rb') as f:
    LSVM_JP_results = pickle.load(f)
    LSVM_JP_results = LSVM_JP_results.round(4)

with open("src/models/results/RF_JP_class_results.pkl", 'rb') as f:
    RF_JP_results = pickle.load(f)
    RF_JP_results = RF_JP_results.round(4)

with open("src/models/results/XGB_JP_class_results.pkl", 'rb') as f:
    XGB_JP_results = pickle.load(f)
    XGB_JP_results = XGB_JP_results.round(4)

### train / test times ###
with open("src/models/results/LG_JP_times.pkl", 'rb') as f:
    LG_JP_times = pickle.load(f)
    LG_JP_times = LG_JP_times.round(4)

with open("src/models/results/LSVC_JP_times.pkl", 'rb') as f:
    LSVM_JP_times = pickle.load(f)
    LSVM_JP_times = LSVM_JP_times.round(4)

with open("src/models/results/RF_JP_times.pkl", 'rb') as f:
    RF_JP_times = pickle.load(f)
    RF_JP_times = RF_JP_times.round(4)

with open("src/models/results/XGB_JP_times.pkl", 'rb') as f:
    XGB_JP_times = pickle.load(f)
    XGB_JP_times = XGB_JP_times.round(4)

accuracy_JP = pd.DataFrame({'Accuracy': [LG_JP_results['accuracy'][1], LSVM_JP_results['accuracy'][1], RF_JP_results['accuracy'][1], XGB_JP_results['accuracy'][1]],
                            'Train Time': [LG_JP_times['train_time'][0], LSVM_JP_times['train_time'][0], RF_JP_times['train_time'][0], XGB_JP_times['train_time'][0]],
                            'Test Time': [LG_JP_times['test_time'][0], LSVM_JP_times['test_time'][0], RF_JP_times['test_time'][0], XGB_JP_times['test_time'][0]]})
accuracy_JP.index = ['Logistic Regression', 'Linear SVM', 'Random Forest', 'XGBoost']
print(accuracy_JP)

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)

table(ax, accuracy_JP)

plt.savefig('visualizations/tables/summary_JP.png',dpi=700, transparent = True, bbox_inches = 'tight', pad_inches = 0)


