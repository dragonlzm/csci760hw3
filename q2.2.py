from myknn import KNNClassifier
import pandas as pd
import numpy as np

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
for i in range(validation_split):
    # split the train and test
    # if i == 0:
    #     train_x = feature[(i+1)*sample_num_in_split:]
    #     train_y = label[(i+1)*sample_num_in_split:]  
    # else:
    train_x = np.concatenate([feature[:(i)*sample_num_in_split], feature[(i+1)*sample_num_in_split:]], axis=0)
    train_y = np.concatenate([label[:(i)*sample_num_in_split], label[(i+1)*sample_num_in_split:]], axis=0)
    test_x = feature[i*sample_num_in_split:(i+1)*sample_num_in_split]
    test_y = label[i*sample_num_in_split:(i+1)*sample_num_in_split]
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    #print(train_x, train_y, test_x, test_y)
    #print(train_x[0].shape, train_y[0].shape, test_x[0].shape, test_y[0].shape)
    #print(train_x[0], train_y[0], test_x[0], test_y[0])
    
    # build the tree
    myknncls = KNNClassifier(1)
    myknncls.train(train_x, train_y)
    
    # do the prediction
    predicted_labels = myknncls.predict(test_x)
    predicted_labels = np.array(predicted_labels)
    
    # calculate the acc, precision and recall
    acc = np.sum(predicted_labels == test_y) / sample_num_in_split
    precision = np.sum((predicted_labels == test_y) & (test_y==1)) / np.sum(predicted_labels==1)
    recall = np.sum((predicted_labels == test_y) & (test_y==1)) / np.sum(test_y==1)
    
    print('Fold:', i+1 ,' acc:', acc, ' precision:', precision, ' recall:', recall)