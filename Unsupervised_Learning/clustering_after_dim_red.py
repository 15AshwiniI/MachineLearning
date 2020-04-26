print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

np.random.seed(42)

# dataset_size = 100

# mnist = fetch_mldata("MNIST original")
# data = scale(mnist.data[:dataset_size]) #/ 255.
# n_digits = len(np.unique(mnist.target))
# labels = mnist.target[:dataset_size]


no_model = False

MNIST = False # Running IRIS or MNIST
PCA = True
RP = False
FA = False
ICA = False
# RP, FA, PCA, ICA


# MNIST
if MNIST:
	from sklearn.datasets import load_digits
	digits = load_digits()
	data = scale(digits.data)
	# n_samples, n_features = data.shape
	# n_digits = len(np.unique(digits.target))
	labels = digits.target

	num_components = 20

	if RP:
		from sklearn.random_projection import GaussianRandomProjection
		model = GaussianRandomProjection(n_components=num_components)

	if FA:
		from sklearn.cluster import FeatureAgglomeration
		model = FeatureAgglomeration(n_clusters=num_components)

	if PCA:
		from sklearn.decomposition import PCA
		model = PCA(n_components=num_components)

	if ICA:
		from sklearn.decomposition import FastICA
		model = FastICA(n_components=num_components)

else:
	# IRIS
	from sklearn.datasets import load_iris
	iris = load_iris()
	data = scale(iris.data)
	# n_samples, n_features = data.shape
	# n_digits = len(np.unique(iris.target))
	labels = iris.target

	num_components = 3

	if RP:
		from sklearn.random_projection import GaussianRandomProjection
		model = GaussianRandomProjection(n_components=num_components)

	if FA:
		from sklearn.cluster import FeatureAgglomeration
		model = FeatureAgglomeration(n_clusters=num_components)

	if PCA:
		from sklearn.decomposition import PCA
		model = PCA(n_components=num_components)

	if ICA:
		from sklearn.decomposition import FastICA
		model = FastICA(n_components=num_components)

if not no_model:
	data = model.fit_transform(data)
	# n_samples, n_features = data.shape






multiple_trials = False

if not multiple_trials:
	print('Applying kmeans to the dataset / Creating the kmeans classifier')
	kmeans = KMeans(n_clusters = num_components, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
	y_kmeans = kmeans.fit_predict(data)
	print metrics.adjusted_rand_score(labels, y_kmeans)

else:
	accuracies = []
	for i in range(1, 15, 1):
	# for i in range(5, 50, 5):
		kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
		y_kmeans = kmeans.fit_predict(data)
		score = metrics.adjusted_rand_score(labels, y_kmeans)
		print("Number of clusters: {}, Score: {}".format(i, score))
		accuracies.append(score)
   	
	print ('Plotting the results onto a line graph, allowing us to observe accuracies')
	plt.plot(range(1, 15, 1), accuracies)
	# plt.plot(range(5, 50, 5), accuracies)
	plt.title('Kmeans accuracies vs number of components')
	plt.xlabel('Number of clusters')
	plt.ylabel('Accuracy (%)') #within cluster sum of squares
	plt.show()