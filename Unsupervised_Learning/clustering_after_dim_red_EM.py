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
	print('Applying EM to the dataset / Creating the gmm classifier')
	gmm = GaussianMixture(n_components = num_components, max_iter = 1000, n_init = 10, random_state = 0, covariance_type='full')
	gmm.fit(data)
	y_gmm = gmm.predict(data)
	print metrics.adjusted_rand_score(labels, y_gmm)

else:
	accuracies = []
	for i in range(5, 50, 5):
		gmm = GaussianMixture(n_components = i, max_iter = 1000, n_init = 10, random_state = 0, covariance_type='full')
		gmm.fit(data)
		y_gmm = gmm.predict(data)
		score = metrics.adjusted_rand_score(labels, y_gmm)
		print("Number of clusters: {}, Score: {}".format(i, score))
		accuracies.append(score)
   	
	print ('Plotting the results onto a line graph, allowing us to observe accuracies')
	plt.plot(range(5, 50, 5), accuracies)
	plt.title('EM accuracies vs number of components')
	plt.xlabel('Number of components')
	plt.ylabel('Accuracy (%)') #within cluster sum of squares
	plt.show()