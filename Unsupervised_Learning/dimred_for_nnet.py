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

PCA = False
RP = True
FA = False
ICA = False

out_file_name = "mnist_digits_rp.csv"
# RP, FA, PCA, ICA

from sklearn.datasets import load_digits
digits = load_digits()
data = scale(digits.data)
labels = digits.target

# MNIST
if not no_model:
	num_components = 15

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

	X_transformed = model.fit_transform(data)
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



