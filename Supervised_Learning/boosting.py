from __future__ import division
from sklearn import ensemble, tree
from sklearn.metrics import mean_squared_error
from sklearn import datasets

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    t = tree.DecisionTreeClassifier(max_depth=8)
    clf = ensemble.AdaBoostClassifier(base_estimator=t, n_estimators=10)

    print 'Boosted Decision tree with dataset size: {}'.format(ds)
    clf = tree.DecisionTreeClassifier(max_depth=8)
    # import pdb; pdb.set_trace()
    clf = clf.fit(train_data, train_labels)

    train_err[i] = mean_squared_error(train_labels, clf.predict(train_data))
    test_err[i] = mean_squared_error(test_labels, clf.predict(test_data))
    print 'Training error: {}'.format(train_err[i])
    print 'Testing Error: '.format(test_err[i])
    print '--------------------------------'

# Plot results
print 'Plot results'
plt.figure()
plt.title('Boosted Decision Trees with MSE vs Dataset Size')
plt.plot(dataset_size, test_err, '-', label='testing error')
plt.plot(dataset_size, train_err, '-', label='training error')
plt.legend()
plt.xlabel('Dataset Size')
plt.ylabel('MSE')
plt.show()