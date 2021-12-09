import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#from sklearn.metrics import confusion_matrix
#with open("src/models/results/LSVC_16_labels.pkl", 'rb') as f:
    #labels = pickle.load(f)

#y_true = labels[0]
y_pred = labels[1]

#conf = confusion_matrix(y_true, y_pred)



conf = [[  0,   1,   0,   1,   0,   0,   0,   0,  16,   8,   1,   0,   0,
          0,   0,   1],
       [  0,  25,   0,   3,   0,   0,   0,   0,  14,  45,   4,   8,   0,
          0,   0,   2],
       [  0,   2,   0,   4,   0,   0,   0,   0,   8,   4,  10,   5,   0,
          0,   0,   2],
       [  0,   4,   0,  26,   0,   0,   0,   0,   9,  17,  14,  31,   0,
          0,   0,   2],
       [  0,   1,   0,   0,   0,   0,   0,   0,   2,   2,   1,   0,   0,
          0,   0,   0],
       [  0,   2,   0,   0,   0,   0,   0,   0,   0,   2,   0,   2,   0,
          0,   0,   1],
       [  0,   0,   0,   0,   0,   0,   0,   0,   3,   0,   2,   0,   0,
          0,   0,   1],
       [  0,   2,   0,   2,   0,   0,   0,   0,   3,   2,   0,   1,   0,
          0,   0,   3],
       [  0,   5,   0,   9,   0,   0,   0,   0,  98,  78,  12,  17,   0,
          0,   0,   2],
       [  0,  10,   0,   6,   0,   0,   0,   0,  35, 195,   6,  22,   0,
          1,   0,   0],
       [  0,   6,   0,   5,   0,   0,   0,   0,  22,  20,  70,  39,   0,
          0,   1,   1],
       [  0,   0,   0,   8,   0,   0,   0,   0,  17,  32,  32, 107,   0,
          0,   0,   0],
       [  0,   2,   0,   0,   0,   0,   0,   0,   5,   9,   2,   6,   0,
          0,   0,   1],
       [  0,   2,   0,   2,   0,   0,   0,   0,   8,  21,   1,   4,   0,
          2,   0,   1],
       [  0,   2,   0,   1,   0,   0,   0,   0,   7,   7,   8,   3,   0,
          0,   3,   0],
       [  0,   2,   0,   2,   0,   0,   0,   0,   2,   8,   2,   5,   0,
          1,   1,  27]]

k = 'ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP','INTJ','INTP','ISFJ','ISFP', 'ISTJ', 'ISTP' #.reverse()
df_cm = pd.DataFrame(conf, index = [i for i in k],
                  columns = [i for i in k])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt='g', cmap="BuPu")
plt.title('LinearSVM Confusion Matrix')
plt.rc('font', size=16)
plt.savefig('lsvc_heatmap.png', dpi=700, bbox_inches = "tight")
plt.show()