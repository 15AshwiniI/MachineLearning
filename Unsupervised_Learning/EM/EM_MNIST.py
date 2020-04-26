print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_mldata
#from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)
print('hi')

dataset_size = 100

# digits = load_digits()

# data = scale(digits.data)
mnist = fetch_mldata("MNIST original")
data = scale(mnist.data[:dataset_size]) #/ 255.
# import pdb; pdb.set_trace()

n_samples, n_features = data.shape
n_digits = len(np.unique(mnist.target))
labels = mnist.target[:dataset_size]

sample_size = 300

# print("n_digits: %d, \t n_samples %d, \t n_features %d"
#       % (n_digits, n_samples, n_features))

wcss = []

print('Applying kmeans to the dataset / Creating the kmeans classifier')
kmeans = GaussianMixture(n_clusters = 10, max_iter = 1000, n_init = 10, random_state = 0, covariance_type='diag')
y_kmeans = kmeans.fit_predict(data)


print('Visualising the clusters')

marker_list = ['red', 'green', 'blue', 'purple', 'magenta', 'orange', 'brown', 'black','cyan', '#33FFA8']
# plt.axis([min(data[:, 0]), max(data[:, 0]), min(data[:, 1]), max(data[:, 1])])
# import pdb; pdb.set_trace()
idx1, idx2 = 130, 215 
for idx in range(10):
  plt.scatter(data[y_kmeans == idx, idx1], data[y_kmeans == idx, idx2], marker = '*', c = marker_list[idx], label = str(idx))

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, idx1], kmeans.cluster_centers_[:, idx2], c = 'yellow', edgecolor = 'black', marker = '*', label = 'Centroids')

plt.legend()
plt.show()

