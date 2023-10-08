import numpy as np
import matplotlib.pyplot as plt
from myknn import KNNClassifier


# load the data
training_data = np.loadtxt('C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw3\\hw3Data\\D2z.txt')
X_train = training_data[:, :-1]  # Features
y_train = training_data[:, -1]   # Labels


# Create a grid of test points
x_range = np.arange(-2, 2.1, 0.1)
y_range = np.arange(-2, 2.1, 0.1)
xx, yy = np.meshgrid(x_range, y_range)
test_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten the grid

# calculate the nearest neighbor
myknn_cls = KNNClassifier(1)
myknn_cls.train(X_train, y_train)

# give prediction
predicted_labels = myknn_cls.predict(test_points)


# Plot training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o', label='Training Points')

# Plot grid points with predicted labels
plt.scatter(test_points[:, 0], test_points[:, 1], c=predicted_labels, cmap=plt.cm.Paired, marker='x', s=20, label='Grid Points')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('1NN Classification with Euclidean Distance')
plt.legend(loc='best')

plt.show()