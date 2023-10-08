import numpy as np
from scipy.spatial.distance import cdist

class KNNClassifier:
    def __init__(self, k=1):
        self.k = k

    # store the number of data
    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train.astype(int)

    def predict(self, X_test, dist='euclidean'):   
        # calculate the pairwise distance
        # assuming the X_test is NxD, and the X_train is MxD
        # the return should be NxM
        distances = cdist(X_test, self.X_train, dist)
        
        # select the idx of smallest index
        topk_smallest_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # test for top1 and top2 
        # top1_val = np.sort(distances, axis=1)[:, 0]
        # top2_val = np.sort(distances, axis=1)[:, 1]
        # first_smallest_indices = np.argsort(distances, axis=1)[:, 0]
        # top1_label = self.y_train[first_smallest_indices]
        # second_smallest_indices = np.argsort(distances, axis=1)[:, 1]
        # top2_label = self.y_train[second_smallest_indices]
        # if np.sum(top1_val == top2_val) > 0:
        #     print("tie num:", np.sum(top1_val == top2_val), \
        #         'top1 label different to the top2 label in tie:', \
        #             np.sum((top1_val == top2_val) & (top1_label != top2_label)))
            
        
        # create a label matrix for the test
        label_matrix = self.y_train[topk_smallest_indices]
        
        # determine the label for each test
        final_label = [np.bincount(label_matrix_row).argmax() for label_matrix_row in label_matrix]
        return final_label
    
    def predict_conf(self, X_test, dist='euclidean'):
        distances = cdist(X_test, self.X_train, dist)
        
        # select the idx of smallest index
        topk_smallest_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # create a label matrix for the test
        label_matrix = self.y_train[topk_smallest_indices]

        confidence = np.sum(label_matrix, axis=-1) / label_matrix.shape[-1]
        return confidence