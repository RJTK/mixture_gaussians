import unittest
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from mixture_gaussians import GaussianMixtureModel


def plot_progress_2D(X, pi, mu, Sigma, Q):
    K = mu.shape[0]
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1])
    for k in range(K):
        ev, EV = np.linalg.eig(Sigma[k])
        el = Ellipse(xy=mu[k, :], width=2 * np.sqrt(5.991 * ev[0]),
                     height=2 * np.sqrt(5.991 * ev[1]),
                     angle=(180. / (np.pi)) * np.arctan(EV[1, 0] /
                                                        EV[0, 0]))
        el.set_alpha(0.5)
        ax.add_artist(el)
        ax.scatter(mu[k, 0], mu[k, 1], marker='X', color='r')
    ax.set_title('%0.5f' % Q)
    plt.show()
    return


class TestMixtureGaussians(unittest.TestCase):
    def test001_basic(self):
        N, p = 150, 2
        X = 2 * np.random.normal(size=(N, p))
        K = 1  # Number of clusters

        gmm = GaussianMixtureModel(num_clusters=K)
        gmm.fit(X)
        return

    def test002_basic(self):
        N, p = 150, 5
        X = np.random.normal(size=(N, p))
        K = 2
        gmm = GaussianMixtureModel(K)
        gmm.fit(X)
        return

    def test003_cluster(self):
        N = 200
        mu = np.array([[-4, 0],
                       [4, 4]])
        Sigma = np.array([[[1., -0.1],
                           [-0.1, 1.3]],

                          [[0.6, 0.2],
                           [0.2, 1.8]]])

        X1 = np.random.multivariate_normal(mu[0], Sigma[0], size=N // 2)
        X2 = np.random.multivariate_normal(mu[1], Sigma[1], size=N // 2)
        X = np.vstack((X1, X2))

        gmm = GaussianMixtureModel(2)
        gmm.fit(X, callback=plot_progress_2D, num_restarts=5)

        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
        x = np.linspace(1.1 * x_min, 1.1 * x_max, 200)
        y = np.linspace(1.1 * y_min, 1.1 * y_max, 200)

        xx, yy = np.meshgrid(x, y)
        xxyy = np.array([xx.ravel(), yy.ravel()]).T
        p = gmm.density(xxyy).reshape(xx.shape)
        plt.scatter(X1[:, 0], X1[:, 1], color='r', marker='x')
        plt.scatter(X2[:, 0], X2[:, 1], color='m', marker='o')
        plt.contourf(x, y, p, alpha=0.5)
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('GMM Density Estimates')
        plt.show()
        return

    def test004_cluster_simple(self):
        N = 200
        mu = np.array([[-2, 0],
                       [2, 2]])
        Sigma = np.array([[[1., -0.1],
                           [-0.1, 1.3]],

                          [[0.6, 0.2],
                           [0.2, 1.8]]])

        X1 = np.random.multivariate_normal(mu[0], Sigma[0], size=N // 2)
        X2 = np.random.multivariate_normal(mu[1], Sigma[1], size=N // 2)
        X = np.vstack((X1, X2))

        gmm = GaussianMixtureModel(2)
        gmm.fit(X, num_restarts=10)
        gmm.plot_scatter_density(X)
        return

    def test005_cluster_complex(self):
        N = 600
        K = 5
        m = 2

        mu = 5 * np.random.normal(size=(K, 2))
        Sigma = np.stack([np.eye(2)] * K)
        for _ in range(m):
            L = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=K)
            Sigma += np.einsum('kp,kq->kpq', L, L)
        Sigma *= 0.5

        X = np.random.multivariate_normal(mu[0], Sigma[0], size=N // 2)
        for k in range(1, K):
            Xk = np.random.multivariate_normal(mu[k], Sigma[k], size=N // 2)
            X = np.vstack((X, Xk))

        gmm = GaussianMixtureModel(K + 3)
        gmm.fit(X, num_restarts=20)
        gmm.plot_scatter_density(X)
        return
