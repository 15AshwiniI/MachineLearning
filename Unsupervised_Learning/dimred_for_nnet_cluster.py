print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

from itertools import product
import random

np.random.seed(42)
output_csv = True

no_model = False

Kmeans = True
EM = False
argmin = True

out_file_name = "mnist_digits_kmeans10.csv"
# RP, FA, PCA, ICA

from sklearn.datasets import load_digits
digits = load_digits()
data = scale(digits.data)
labels = digits.target

# MNIST
if not no_model:
	num_components = 10

	if Kmeans:
		from sklearn.cluster import KMeans
		model = KMeans(n_clusters = num_components, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
		X_transformed = model.fit_transform(data)
		if argmin:
			X_transformed = np.argmin(X_transformed, axis=1)
			X_transformed = np.reshape(X_transformed, [-1, 1])
	if EM:
		from sklearn.mixture import GaussianMixture
		model = GaussianMixture(n_components = num_components, max_iter = 1000, n_init = 10, random_state = 0, covariance_type='full')
		model.fit(data)
		X_transformed = model.predict(data)
		X_transformed = np.reshape(X_transformed, [-1, 1])

else:
	X_transformed = data

if output_csv:
	print X_transformed.shape
	with open(out_file_name, "w") as outfile:
		outfile.write('Header\n')
		for idx, sample in enumerate(X_transformed):
			output_str=""
			for point in sample:
				output_str += str(point) + ","
			output_str += str(int(labels[idx])) + "\n"
			outfile.write(output_str)



