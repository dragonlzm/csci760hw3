from myknn import KNNClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

validation_split = 5

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw3\\hw3Data\\emails.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# get the feature and label
feature = df.iloc[:, 1:3001].to_numpy()
label = df.iloc[:, 3001].to_numpy()

# do the cross validation
num_data = feature.shape[0]
sample_num_in_split = int(num_data / validation_split)
acc_list = []
precision_list = []
recall_list = []

for k in [1,3,5,7,10]:
    local_acc = []
    local_precision = []
    local_recall = []
    for i in range(validation_split):
        train_x = np.concatenate([feature[:(i)*sample_num_in_split], feature[(i+1)*sample_num_in_split:]], axis=0)
        train_y = np.concatenate([label[:(i)*sample_num_in_split], label[(i+1)*sample_num_in_split:]], axis=0)
        test_x = feature[i*sample_num_in_split:(i+1)*sample_num_in_split]
        test_y = label[i*sample_num_in_split:(i+1)*sample_num_in_split]
        
        # build the tree
        myknncls = KNNClassifier(k)
        myknncls.train(train_x, train_y)
        
        # do the prediction
        predicted_labels = myknncls.predict(test_x)
        predicted_labels = np.array(predicted_labels)
        
        # calculate the acc, precision and recall
        acc = np.sum(predicted_labels == test_y) / sample_num_in_split
        precision = np.sum((predicted_labels == test_y) & (test_y==1)) / np.sum(predicted_labels==1)
        recall = np.sum((predicted_labels == test_y) & (test_y==1)) / np.sum(test_y==1)
        
        local_precision.append(precision)
        local_acc.append(acc)
        local_recall.append(recall)
        print('Fold:', i+1 ,' acc:', acc, ' precision:', precision, ' recall:', recall)
    acc_list.append(sum(local_acc)/len(local_acc))
    precision_list.append(sum(local_precision)/len(local_precision))
    recall_list.append(sum(local_recall)/len(local_recall))

print('acc_list:', acc_list)
print('precision_list:', precision_list)
print('recall_list:', recall_list)

# acc_list: [0.8344000000000001, 0.841, 0.8418000000000001, 0.8452, 0.8558]
# precision_list: [0.6784185058833994, 0.6957001908265781, 0.7007718233382614, 0.707131120317517, 0.7455679622371316]
# recall_list: [0.8220387439573147, 0.8115969196051873, 0.8029117082298332, 0.8080797222641802, 0.771662824404493]

#acc_list = [0.8344000000000001, 0.841, 0.8418000000000001, 0.8452, 0.8558]

plt.scatter([1,3,5,7,10], acc_list, marker='o', c='b')
plt.plot([1,3,5,7,10], acc_list)
plt.title('KN 5-fold Cross Validation')
plt.xlabel('k')
plt.ylabel('Average accuracy')
plt.legend()
plt.grid(True)
plt.show()
