import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import FastICA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import random

sample_before = True
plot = True
num_components = 2
output_csv = False
#use all digits
dataset_size = 1000
smaller_dataset = True
PCA = True

if smaller_dataset:
	from sklearn.datasets import load_digits
	mnist = load_digits()
	X_train, y_train = mnist.data/255. , mnist.target
else:
	out_file_name = "mnist_pca_after.csv"

	mnist = fetch_mldata("MNIST original")
	X_train, y_train = mnist.data[:70000] / 255., mnist.target[:70000]

# if sample_before:
if not sample_before:
	X_train_sample = []
	y_train_sample = []
	for idx in random.sample(xrange(70000), dataset_size):
		X_train_sample.append(X_train[idx])
		y_train_sample.append(y_train[idx])

	X_train = X_train_sample
	y_train = y_train_sample

# import ipdb; ipdb.set_trace()
# X_train, y_train = random.sample(zip(mnist.data[:70000] / 255., mnist.target[:70000]), dataset_size)

#X_train, y_train = shuffle(X_train, y_train)
#X_train, y_train = X_train[:1000], y_train[:1000]  # lets subsample a bit for a first impression
if not PCA:
	from sklearn.decomposition import FastICA
	pca = FastICA(n_components=num_components)
else:
	from sklearn.decomposition import PCA
	pca = PCA(n_components=num_components)
if plot:
	fig, plot = plt.subplots()
	fig.set_size_inches(50, 50)
	plt.prism()

X_transformed = pca.fit_transform(X_train)

if plot:
	plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_train)
	plot.set_xticks(())
	plot.set_yticks(())

	plt.tight_layout()
	plt.savefig("mnist_ica.png")

if not sample_before:
	X_train_sample = []
	y_train_sample = []
	for idx in random.sample(xrange(70000), dataset_size):
		X_train_sample.append(X_transformed[idx])
		y_train_sample.append(y_train[idx])

	# import ipdb; ipdb.set_trace()
	X_transformed = X_train_sample
	y_train = y_train_sample

if output_csv:
	with open(out_file_name, "w") as outfile:
		for idx, sample in enumerate(X_transformed):
			output_str=""
			for point in sample:
				output_str += str(point) + ","
			output_str += str(int(y_train[idx])) + "\n"
			outfile.write(output_str)