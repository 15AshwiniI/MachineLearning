import numpy as np
import pickle
import sys

from sklearn.decomposition import PCA

# From https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def to_grayscale(data):
	return_value = []
	for data_point in data:
		grayed_data_point = []
		for R, G, B in zip(data_point[:1024], data_point[1024:2048], data_point[2048:]):
			grayed_data_point.append(0.299*R + 0.587*G + 0.114*B)
		return_value.append(grayed_data_point)
	return return_value

def eu_distance(x, y):
	return np.sqrt(sum([(xdim-ydim)**2 for xdim, ydim in zip(point1, point2)]))

def nearest_neighbors(training_data, training_data_labels, test_sample, K):
	dists = [eu_distance(data, test_sample) for data in training_data]
	dist_arr = np.array(dists)
	knn_indices = dist_arr.argsort()[:K]
	voted_label = {}
	for k_index in knn_indices:
		voting_weight = 1.0/eu_distance(training_data[k_index], test_sample)
		k_label = training_data_labels[k_index]
		if k_label in voted_label:
			voted_label[k_label] += voting_weight
		else:
			voted_label[k_label] = voting_weight
	maxVal = 0
	voted_key = None
	for key, val in voted_label.items():
		if val > maxVal:
			voted_key = key
			maxVal = val
	return voted_key


if __name__ == '__main__':
	K, D, N, data_path = sys.argv[1:5]
	data_dict = unpickle(data_path)
	labels = data_dict['labels'.encode('UTF-8')]
	data = data_dict['data'.encode('UTF-8')]
	sub_labels = labels[:1000]
	sub_batch = data[:1000]

	grayed_data = to_grayscale(sub_batch)

	test_set_labels = sub_labels[:N]
	test_set = grayed_data[:N]

	training_set_labels = sub_labels[N:]
	training_set = grayed_data[N:]
	# PCA
	pca = PCA(n_components=D, svd_solver='full')
	pca.fit(training_set)

	reduced_training_data = pca.transform(training_set)
	reduced_test_data = pca.transform(test_set)
