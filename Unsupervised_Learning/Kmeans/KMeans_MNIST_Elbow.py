print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_digits
#from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)
print('hi')

# dataset_size = 10000

digits = load_digits()

data = scale(digits.data)
# mnist = fetch_mldata("MNIST original")
# data = scale(mnist.data[:dataset_size]) #/ 255.
#import pdb; pdb.set_trace()

n_samples, n_features = data.shape
# n_digits = len(np.unique(mnist.target))
# labels = mnist.target[:dataset_size]

wcss = []

for i in range(5, 200, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(data)
    print(kmeans.inertia_)
    wcss.append(kmeans.inertia_)
    
print ('Plotting the results onto a line graph, allowing us to observe The elbow')
plt.plot(range(5, 200, 20), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster Sum of Squares') #within cluster sum of squares
plt.show()