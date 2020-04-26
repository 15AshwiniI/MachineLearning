I used the MNIST and CIFAR 10 datasets. The download urls are as follows:
MNIST - http://yann.lecun.com/exdb/mnist/
CIFAR - https://www.cs.toronto.edu/~kriz/cifar.html

As only the preprocessing for loading the data differs from the datasets, the loading of data for each file is
separated while the classifier code is identical for most classifiers. By default, CIFAR is commented out.

I primarily used the sklearn library and tensorflow for neural networks.
I've implemented my own version of knn and also tried the benchmarked sklearn version.

the cifar10*.py files are for running convolutional neural networks on the Cifar10 dataset and come from the 
Tensorflow tutorial. 
The MNIST CNN code is taken from Krzysztof Furman and some eval code was used from the Tensorflow tutorial.

The training and test data come from the same data source and are split and handled in code (they aren’t physically separated)

Libraries required - matplotlib, tensorflow, sklearn