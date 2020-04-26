import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class RandomProjector(object):
    def __init__(self, X, y, frames=200):
        self.X = X
        self.y = y
        self.frames = frames

        if y is not None:
            # Spectral has 255 colors (I think)
            num_classes = len(np.unique(y))
            colors = plt.cm.Spectral(np.arange(num_classes)
                                     * 255 / num_classes)
        else:
            num_classes = 1
            colors = 'b'

        self.points = [plt.plot([], [], 'o', color=colors[i])[0]
                  for i in range(num_classes)]

        n_features = X.shape[1]

        # initialize projection matrix
        self.projection = np.zeros((n_features, 2))

        # rest is n_features - 2 large
        size = n_features - 2
        self.frequencies = 10 + np.random.randint(10, size=(size, 2))
        self.phases = np.random.uniform(size=(size, 2))

    def init_figure(self):
        for p in self.points:
            p.set_data([], [])
        plt.xlim((-2, 2))
        plt.ylim((-2, 2))
        plt.xticks(())
        plt.yticks(())
        return self.points

    def animate(self, i):
        # set top 2x2 to identity
        self.projection[0, 0] = 1
        self.projection[1, 1] = 1
        # set "free entries" of projection matrix
        # gives them a "rotation" feel and makes the whole thing seamless.
        scale = 2 * np.pi * i / self.frames
        self.projection[2:, :] = np.sin(self.frequencies * scale + self.phases)
        interpolation = np.dot(X, self.projection)
        interpolation /= interpolation.max(axis=0)
        for p, c in zip(self.points, np.unique(y)):
            p.set_data(interpolation[y == c, 0], interpolation[y == c, 1])
        return self.points


def make_video(X, y=None, frames=500, filename="video.mp4"):
    fig = plt.figure()
    projector = RandomProjector(X, y, frames)
    anim = FuncAnimation(fig, projector.animate, frames=frames, interval=100,
            blit=True, init_func=projector.init_figure)

    #anim.save(filename, fps=20, extra_args=['-vcodec', 'libx264'])

    plt.show()


if __name__ == "__main__":
    iris = load_iris()
    #iris = load_digits()
    X, y = iris.data, iris.target

    mask = (y == 1) + (y == 2) + (y == 7)
    y = y[mask]
    X = X[mask]

    # we should at least remove the mean
    X = StandardScaler(with_std=False).fit_transform(X)

    # make boring PCA visualization for comparison
    num_classes = len(np.unique(y))
    colors = plt.cm.Spectral(np.arange(num_classes) * 255 / num_classes)
    X_pca = PCA(n_components=2).fit_transform(X)
    for i, c in enumerate(np.unique(y)):
        plt.plot(X_pca[y == c, 0], X_pca[y == c, 1], 'o', color=colors[i],
                label=c)
    plt.legend()
    plt.savefig("digits_pca.png", bbox_inches="tight")
    # PCA here optional. Also try without.
    X = PCA().fit_transform(X)
    make_video(X, y, filename='digits_two_classes.mp4', frames=1000)
# import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler

# iris = load_iris()
# X, y = iris.data, iris.target

# X_pca = StandardScaler().fit_transform(X)

# fig = plt.figure()
# colors = plt.cm.Spectral(y * 255 / y.max())

# n_iter = 200

# points = [plt.plot([], [], 'o', color=['r', 'g', 'b'][i])[0]
#           for i in np.unique(y)]


# def init():
#     global points
#     for p in points:
#         p.set_data([], [])
#     plt.xlim((-4, 4))
#     plt.ylim((-3, 3))
#     plt.xticks(())
#     plt.yticks(())
#     return points


# def animate(i):
#     global points
#     alpha = 2 * np.pi * i / n_iter
#     beta = 4 * np.pi * i / n_iter
#     interpolation1 = np.cos(alpha) * X_pca[:, 1] + np.sin(alpha) * X_pca[:, 2]
#     interpolation2 = np.cos(beta) * X_pca[:, 0] + np.sin(beta) * X_pca[:, 3]
#     for p, c in zip(points, np.unique(y)):
#         p.set_data(interpolation1[y == c], interpolation2[y == c])
#     return points

# anim = FuncAnimation(fig, animate, frames=n_iter, interval=100, blit=True,init_func=init)
# #anim.save("iris.mp4", fps=20, extra_args=['-vcodec', 'libx264'])
# plt.show()