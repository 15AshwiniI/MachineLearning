# #Not working, can you help me?

# from sklearn.cluster import FeatureAgglomeration
# import pandas as pd
# import matplotlib.pyplot as plt

# #iris.data from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
# iris=pd.read_csv('Iris.csv',sep=',',header=None)
# #store labels
# label=iris[4]
# iris=iris.drop([4],1)

# #set n_clusters to 2, the output will be two columns of agglomerated features ( iris has 4 features)
# agglo=FeatureAgglomeration(n_clusters=2).fit_transform(iris)

# #plotting
# color=[]
# for i in label:
#     if i=='Iris-setosa':
#         color.append('g')
#     if  i=='Iris-versicolor':
#         color.append('b')
#     if i=='Iris-virginica':
#         color.append('r')
# plt.scatter(agglo[:,0],agglo[:,1],c=color)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
from sklearn.cluster import FeatureAgglomeration

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
pca = FeatureAgglomeration(n_clusters=3)
pca.fit(X)
X = pca.transform(X)

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