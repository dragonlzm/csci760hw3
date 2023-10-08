
import pandas as pd
import numpy as np
from my_logistic_regression import MyLogisticRegression

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

lr = 0.0005, 
train_iter = 5000


for i in range(validation_split):

    train_x = np.concatenate([feature[:(i)*sample_num_in_split], feature[(i+1)*sample_num_in_split:]], axis=0)
    train_y = np.concatenate([label[:(i)*sample_num_in_split], label[(i+1)*sample_num_in_split:]], axis=0)
    test_x = feature[i*sample_num_in_split:(i+1)*sample_num_in_split]
    test_y = label[i*sample_num_in_split:(i+1)*sample_num_in_split]

    # do the normalization of the x
    #train_x = train_x / np.linalg.norm(train_x, axis=1)[:, np.newaxis]
    #test_x = test_x / np.linalg.norm(test_x, axis=1)[:, np.newaxis]
    
    #  initial the model parameters
    my_lg_cls = MyLogisticRegression(train_x.shape[-1], lr, train_iter)

    # train the model
    my_lg_cls.train(train_x, train_y, test_x, test_y)
    
    # make prediction
    predicted_labels = my_lg_cls.predict(test_x)
    acc = np.sum(predicted_labels == test_y) / sample_num_in_split
    precision = np.sum((predicted_labels == test_y) & (test_y==1)) / np.sum(predicted_labels==1)
    recall = np.sum((predicted_labels == test_y) & (test_y==1)) / np.sum(test_y==1)
    
    print('Fold:', i+1 ,' acc:', acc, ' precision:', precision, ' recall:', recall)
    #break
