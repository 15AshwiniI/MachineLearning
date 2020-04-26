from __future__ import division
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

""" Out of library implementation is at bottom
"""


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# num_sample = mnist.train.num_examples
# input_dim = mnist.train.images[0].shape[0]
# w = h = 28

def load_mnist(data_dir='MNIST_data', flatten=False, one_hot=True, normalize_range=False):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(data_dir, one_hot=one_hot, reshape=flatten)

    def _extract_fn(x):
        X = np.reshape(x.images, [x.images.shape[0], -1])
        y = np.argmax(x.labels, axis=1)

        if not normalize_range: 
            X *= 255.0
        
        return (X, y)
        
    Xtrain, ytrain = _extract_fn(mnist.train)
    Xval, yval = _extract_fn(mnist.validation)
    Xtest, ytest = _extract_fn(mnist.test)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest) 

Xtrain, ytrain, Xval, yval, Xtest, ytest = load_mnist()

# For CIFAR, one batch
# train_batch = cPickle.load(open('datasets/cifar-10-batches-py/data_batch_1', 'rb'))
# test_batch = cPickle.load(open('datasets/cifar-10-batches-py/test_batch', 'rb'))

# # Each batch has 10k samples

# train_data = train_batch['data'] # 2D array [samples, 32*32]
# test_data = test_batch['data']
# train_labels = train_batch['labels'] # 1d array [samples]
# test_labels = test_batch['labels']
# train_size = len(train_data)

# offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
# train_err = [0] * len(offsets)
# test_err = [0] * len(offsets)

# import pdb; pdb.set_trace()
dataset_size = range(1000, len(Xtest), len(Xtest) // 10)
train_err = [0] * len(dataset_size)
test_err = [0] * len(dataset_size)

for i, ds in enumerate(dataset_size):
    train_data, test_data, train_labels, test_labels = Xtrain[:ds], Xtest[:ds], ytrain[:ds], ytest[:ds]
    # train_data, test_data, train_labels, test_labels = train_test_split(X[:b], Ytrain[:b], test_size=0.20, random_state=42)
    # import pdb; pdb.set_trace()
    # sentiment_data = train_data[:b]
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)

    print 'KNN with dataset size: {}'.format(ds)
    clf = clf.fit(train_data, train_labels)

    train_err[i] = mean_squared_error(train_labels, clf.predict(train_data))
    test_err[i] = mean_squared_error(test_labels, clf.predict(test_data))
    print 'Training error: {}'.format(train_err[i])
    print 'Testing Error: '.format(test_err[i])
    print '--------------------------------'


# Plot results
print 'Plot results'
plt.figure()
plt.title('KNN with MSE vs Dataset Size')
plt.plot(dataset_size, test_err, '-', label='testing error')
plt.plot(dataset_size, train_err, '-', label='training error')
plt.legend()
plt.xlabel('Dataset Size')
plt.ylabel('MSE')
plt.show()


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