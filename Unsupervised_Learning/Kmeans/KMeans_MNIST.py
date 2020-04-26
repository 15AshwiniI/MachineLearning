print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
# from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_digits
#from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

# dataset_size = 100

# mnist = fetch_mldata("MNIST original")
# data = scale(mnist.data[:dataset_size]) #/ 255.
# n_digits = len(np.unique(mnist.target))
# labels = mnist.target[:dataset_size]

digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target



multiple_trials = True

if not multiple_trials:
	print('Applying kmeans to the dataset / Creating the kmeans classifier')
	kmeans = KMeans(n_clusters = 20, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
	y_kmeans = kmeans.fit_predict(data)
	print metrics.adjusted_rand_score(labels, y_kmeans)

else:
	accuracies = []
	for i in range(5, 50, 5):
		kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
		y_kmeans = kmeans.fit_predict(data)
		score = metrics.adjusted_rand_score(labels, y_kmeans)
		print("Number of clusters: {}, Score: {}".format(i, score))
		accuracies.append(score)
   	
	print ('Plotting the results onto a line graph, allowing us to observe accuracies')
	plt.plot(range(5, 50, 5), accuracies)
	plt.title('Kmeans accuracies vs number of components')
	plt.xlabel('Number of clusters')
	plt.ylabel('Accuracy (%)') #within cluster sum of squares
	plt.show()



# Can't visualize because data is too high dimensional
# print('Visualising the clusters')

# marker_list = ['red', 'green', 'blue', 'purple', 'magenta', 'orange', 'brown', 'black','cyan', '#33FFA8']
# # plt.axis([min(data[:, 0]), max(data[:, 0]), min(data[:, 1]), max(data[:, 1])])
# # import pdb; pdb.set_trace()
# idx1, idx2 = 130, 215 
# for idx in range(10):
#   plt.scatter(data[y_kmeans == idx, idx1], data[y_kmeans == idx, idx2], marker = '*', c = marker_list[idx], label = str(idx))

# #Plotting the centroids of the clusters
# plt.scatter(kmeans.cluster_centers_[:, idx1], kmeans.cluster_centers_[:, idx2], c = 'yellow', edgecolor = 'black', marker = '*', label = 'Centroids')

# plt.legend()
# plt.show()
