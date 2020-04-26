#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import scale

print 'hello'
from sklearn import metrics
from sklearn.mixture import GaussianMixture

np.random.seed(42)

digits = datasets.load_iris()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target


multiple_trials = True

if not multiple_trials:
	print('Applying EM to the dataset / Creating the gmm classifier')
	gmm = GaussianMixture(n_components = 25, max_iter = 1000, n_init = 10, random_state = 0, covariance_type='full')
	gmm.fit(data)
	y_gmm = gmm.predict(data)
	print metrics.adjusted_rand_score(labels, y_gmm)

else:
	accuracies = []
	for i in range(1, 20, 2):
		gmm = GaussianMixture(n_components = i, max_iter = 1000, n_init = 10, random_state = 0, covariance_type='full')
		gmm.fit(data)
		y_gmm = gmm.predict(data)
		score = metrics.adjusted_rand_score(labels, y_gmm)
		print("Number of clusters: {}, Score: {}".format(i, score))
		accuracies.append(score)
   	
	print ('Plotting the results onto a line graph, allowing us to observe accuracies')
	plt.plot(range(1, 20, 2), accuracies)
	plt.title('EM accuracies vs number of components')
	plt.xlabel('Number of components')
	plt.ylabel('Accuracy (%)') #within cluster sum of squares
	plt.show()

