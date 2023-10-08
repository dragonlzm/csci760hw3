import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from my_logistic_regression import MyLogisticRegression
from myknn import KNNClassifier
from sklearn.metrics import auc, roc_curve


# def get_tpr_fpr_list(pred_conf, gt_label):
#     num_pos = np.sum(gt_label == True).item()
#     num_neg = np.sum(gt_label == False).item()
#     tp = 0
#     fp = 0
#     last_tp = 0
#     fpr_list = [0]
#     tpr_list = [0]
#     for i in range(len(pred_conf)):
#         if (i > 0) and (gt_label[i] != gt_label[i-1]) and (gt_label[i] == False) \
#             and tp > last_tp:
#                 last_tp = tp
#                 tpr = tp / num_pos
#                 fpr = fp / num_neg
#                 tpr_list.append(tpr)
#                 fpr_list.append(fpr)
#         if gt_label[i] == True:
#             tp += 1
#         else:
#             fp += 1
#     tpr = tp / num_pos
#     fpr = fp / num_neg
#     tpr_list.append(tpr)
#     fpr_list.append(fpr)
#     return tpr_list, fpr_list

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw3\\hw3Data\\emails.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# get the feature and label
feature = df.iloc[:, 1:3001].to_numpy()
label = df.iloc[:, 3001].to_numpy()
train_x = feature[:4000]
train_y = label[:4000]
test_x = feature[4000:]
test_y = label[4000:]

###### for logistic regression initial the model parameters
lr = 0.0005
train_iter = 5000
my_lg_cls = MyLogisticRegression(train_x.shape[-1], lr, train_iter)

# train the model
my_lg_cls.train(train_x, train_y, test_x, test_y)

# make prediction (condf)
lg_pred_conf = my_lg_cls.sigmoid(test_x.dot(my_lg_cls.param))
# do the plotting
#lg_tpr_list, lg_fpr_list = get_tpr_fpr_list(lg_pred_conf, test_y)
lg_fpr_list, lg_tpr_list , _ = roc_curve(test_y, lg_pred_conf)
roc_auc_lg = auc(lg_fpr_list, lg_tpr_list)


########### for decision tree
# build the tree
myknncls = KNNClassifier(5)
myknncls.train(train_x, train_y)

# do the prediction
knn_pred_conf = myknncls.predict_conf(test_x)

knn_fpr_list, knn_tpr_list , _ = roc_curve(test_y, knn_pred_conf)
roc_auc_knn = auc(knn_fpr_list, knn_tpr_list)

#plt.figure(figsize=(8, 6))
plt.plot(lg_fpr_list, lg_tpr_list, label='logistic regression (AUC = %0.2f)' % roc_auc_lg)
plt.plot(knn_fpr_list, knn_tpr_list, label='knn (AUC = %0.2f)' % roc_auc_knn)
plt.title('roc curve')
plt.xlabel('FP rate')
plt.ylabel('TP rate')
plt.legend()
plt.grid(True)
plt.show()
