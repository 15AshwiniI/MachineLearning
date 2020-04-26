import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale


from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
orig_X = X
y = iris.target
data = scale(iris.data)
n_samples, n_features = data.shape

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
# import ipdb; ipdb.set_trace()
# print pca.explained_variance_

# n_samples = X.shape[0]
# We center the data and compute the sample covariance matrix.
# X -= np.mean(X, axis=0)
# cov_matrix = np.dot(X.T, X) / n_samples
# import ipdb; ipdb.set_trace()
# for eigenvector in pca.components_:
#     print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
X_centered = orig_X - np.mean(orig_X, axis=0)
eigenvalues = pca.explained_variance_
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print(eigenvalue)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, #cmap=plt.cm.spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()