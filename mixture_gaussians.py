"""
Implements EM algorithm for fitting a mixture of gaussian densities.
"""
import logging

import numpy as np
import matplotlib.colors as colors

from matplotlib import pyplot as plt
import seaborn

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Data dimensions:
# K = num_clusters
# X: N x p -- data matrix
# mu: K x p -- Mean for each cluster
# Sigma: K x p x p -- Variance for each cluster
# pi: K -- Cluster assignment probabilities sum(pi) = 1.0
# r: K x N -- P(z=k|x) responsibility cluster k takes for x


class EstepException(Exception):
    pass


class QstepException(Exception):
    pass


class MstepException(Exception):
    pass


class GaussianMixtureModel:
    def __init__(self, num_clusters, do_logging=False, prune_clusters=True,
                 MAP_regularize=False):
        """
        A Gaussian mixture model.

        params:
          num_clusters (int): Number of clusters to fit to the data
          do_logging (bool): Whether or not to log debug events
          prune_clusters (bool): If True, we will prune clusters with a
            nearly singular covariance or with small mixture weights.  This
            is slower, but improves numerical stability.  It also means that
            the resulting number of clusters may be less than num_clusters.
          MAP_regularize (bool): If True, we will add a standard regularizer,
            or equivalently perform MAP estimation.  This greatly improves
            the numerical robustness and is more or less essential in high
            dimensions.
        """
        self.num_clusters = num_clusters
        self._fitted = False
        # If a cluster has near 0 weight or a near singular variance
        # we will prune this cluster and reduce the number of clusters
        # by 1.  This should never reduce the number of clusters to 0
        # except in extremely degenerate cases
        self.prune_clusters = prune_clusters
        self.MAP_regularize = MAP_regularize
        self.do_logging = do_logging
        if self.do_logging:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler('mixture_gaussians.log')
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.debug("----------- Fitting Gaussian "
                              "Mixture Model --------------")
        return

    def _Estep(self, X, pi, mu, Sigma):
        U = np.moveaxis(X[:, None, :] - mu, 0, 1)
        try:
            log_Px = -0.5 * (np.einsum('kip,kpq,kiq->ki',
                                       U, np.linalg.inv(Sigma), U)
                             + np.linalg.slogdet(Sigma)[1][:, None])
        except np.linalg.LinAlgError as e:
            if self.do_logging:
                self.logger.error('LinAlgError in Estep: %s' % e)
            raise EstepException('LinAlgError')

        M = np.max(log_Px, axis=0)
        logsum_piPx = M + np.log(np.einsum('k,ki->i', pi,
                                           np.exp(log_Px - M)))
        r = (pi[:, None] * np.exp(log_Px)) / np.exp(logsum_piPx)
        if np.any(np.isnan(r)):
            if self.do_logging:
                self.logger.error('r contains nan r = %s, pi = %s\n'
                                  'Setting nans to 0 and continuing' % (r, pi))
            r[np.isnan(r)] = 0

        return r

    def _Mstep(self, X, r):
        N = X.shape[0]
        pi = (1. / N) * np.sum(r, axis=1)
        mu = np.einsum('ki,ip->kp', r, X) / (N * pi[:, None])
        Sigma = (np.einsum('ki,ip,iq->kpq',
                           r, X, X) / (N * pi[:, None, None]) -
                 np.einsum('kp,kq->kpq', mu, mu))
        return pi, mu, Sigma

    def _Mstep_MAP(self, X, r):
        S0, K = self.S0, self.num_clusters
        N = X.shape[0]  # Number of samples
        p = X.shape[1]  # Dimension

        pi = (1. / N) * np.sum(r, axis=1)
        mu = np.einsum('ki,ip->kp', r, X) / (N * pi[:, None])
        U = np.moveaxis(X[:, None, :] - mu, 0, 1)
        S = np.einsum('ki,kip,kiq->kpq', r, U, U)
        S0 = S0 / (K ** (1. / p))
        S[:, range(p), range(p)] += S0
        Sigma = S / (2 * p + 4 + N * pi)[:, None, None]
        return pi, mu, Sigma

    def _Qfunc(self, X, r, pi, mu, Sigma):
        p = X.shape[1]
        log_pi = np.log(pi)
        if np.any(np.isnan(log_pi)):
            if self.do_logging:
                self.logger.error('nan in log_pi.  pi = %s' % pi)
            raise QstepException('nan in log_pi')

        cluster_quality = np.sum(log_pi @ r)
        U = np.moveaxis(X[:, None, :] - mu, 0, 1)

        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError as e:
            if self.do_logging:
                self.logger.error('LinAlgError in Q: %s' % e)
            raise QstepException('LinAlgError')

        nll = -np.dot(np.sum(r, axis=1),
                      np.sum(np.log(L[:, range(p), range(p)]), axis=1))
        Z = np.einsum('kpq,kiq->kip', np.linalg.inv(L), U)
        nll -= 0.5 * np.einsum('ki,ki', r, np.linalg.norm(Z, axis=2)**2)

        Q = cluster_quality + nll
        return Q

    def _prune(self, r, pi, mu, Sigma):
        k = 0
        while k < mu.shape[0]:
            s_min = np.min(np.linalg.svd(Sigma[k], compute_uv=False))
            if pi[k] < 1e-3 or s_min < 1e-3:
                pi = np.delete(pi, k)
                mu = np.delete(mu, k, axis=0)
                Sigma = np.delete(Sigma, k, axis=0)
                r = np.delete(r, k, axis=0)
                if self.do_logging:
                    self.logger.warning('Pruned a cluster!')
            else:
                k += 1
        return r, pi, mu, Sigma

    def fit(self, X, num_restarts=1, max_steps=50,
            eps=1e-6, debug=False, callback=None):
        N, p = X.shape
        K = self.num_clusters
        if self.MAP_regularize:
            self.S0 = np.var(X, axis=0)

        Q_best = -np.inf
        for _ in range(num_restarts):
            if self.do_logging:
                self.logger.info('----- EM restart ------')

            # Random initialization
            mu = np.random.normal(size=(K, p))
            Sigma = np.stack([np.eye(p)] * K)
            pi = np.abs(np.random.normal(size=K))
            pi = pi / np.sum(pi)

            # Intialization for loop conditions
            Q_prev = -np.inf
            delta_Q = np.inf
            step_count = 0

            while delta_Q > eps and step_count < max_steps:
                step_count += 1
                try:
                    r = self._Estep(X, pi, mu, Sigma)
                except EstepException as e:
                    if self.do_logging:
                        self.logger.error('Caught EstepException %s.  '
                                          'Continuing on next random restart'
                                          % e)
                    break

                try:
                    if not self.MAP_regularize:
                        pi, mu, Sigma = self._Mstep(X, r)
                    else:
                        pi, mu, Sigma = self._Mstep_MAP(X, r)
                except MstepException as e:
                    if self.do_logging:
                        logging.error('Caught MstepException %s.  '
                                      'Continuing on next random restart' % e)
                    break

                if self.prune_clusters:
                    r, pi, mu, Sigma = self._prune(r, pi, mu, Sigma)

                try:
                    Q = self._Qfunc(X, r, pi, mu, Sigma)
                except QstepException as e:
                    if self.do_logging:
                        self.logger.error('Caught QstepException %s.  '
                                          'Continuing on next random restart'
                                          % e)
                    break

                delta_Q = Q - Q_prev
                Q_prev = Q
                if self.do_logging:
                    self.logger.debug("Step %d: Q = %0.5f, delta_Q = %0.5f"
                                      % (step_count, Q, delta_Q))
                    if delta_Q < -np.abs(eps):
                        self.logger.error("Q did not decrease!")

            if Q > Q_best:
                Q_best = Q
                pi_best, mu_best, Sigma_best = pi, mu, Sigma

            if callback is not None:
                callback(X, pi_best, mu_best, Sigma_best, Q_best)

            if self.do_logging:
                self.logger.debug('Done.  pi_best = %s, mu_best = %s, '
                                  'Sigma_best = %s'
                                  % (pi_best, mu_best, Sigma_best))

        if callback is not None:
            callback(X, pi_best, mu_best, Sigma_best, Q_best)

        self.pi, self.mu, self.Sigma = pi_best, mu_best, Sigma_best
        self.Q_best = Q_best
        if self.prune_clusters:
            K = self.mu.shape[0]
            if K < self.num_clusters:
                self.num_clusters = K
                if self.do_logging:
                    self.logger.warning('Max Q Cluster count is less than '
                                        'the initialization.  K = %d' % K)
        self._fitted = True
        return

    def density(self, xxyy):
        """Evaluate the fitted density at the points X, an (M x p) array"""
        if not self._fitted:
            raise AssertionError('Model not fit')
        pi, mu, Sigma = self.pi, self.mu, self.Sigma
        U = np.moveaxis(xxyy[:, None, :] - mu, 0, 1)
        try:
            log_Px = -0.5 * (np.einsum('kip,kpq,kiq->ki',
                                       U, np.linalg.inv(Sigma), U)
                             + np.linalg.slogdet(Sigma)[1][:, None])
            # This will fail if Sigma is a singular matrix
        except np.linalg.LinAlgError as e:
            self.logger.error('LinAlgError in density(): %s' % e)
            raise e

        M = np.max(log_Px, axis=0)
        logsum_piPx = M + np.log(np.einsum('k,ki->i', pi,
                                           np.exp(log_Px - M)))
        p = np.exp(logsum_piPx)
        if np.any(np.isnan(p)):
            if self.do_logging:
                self.logger.error('density p contains nan p = %s, log_p = %s\n'
                                  'Setting nans to 0 and continuing'
                                  % (p, log_Px))
            p[np.isnan(p)] = 0
        return p

    def plot_scatter_density(self, X, fit=True):
        """Plots the first 2 dimensions of data X and the density estimate
        of the GMM"""
        if not self._fitted:
            if fit:
                self.fit(X)
            else:
                raise AssertionError('Model not fit and specified fit=False')

        if X.shape[1] > 2:
            raise ValueError("We are only able to plot in 2 dimensions")
        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
        x = np.linspace(1.1 * x_min, 1.1 * x_max, 350)
        y = np.linspace(1.1 * y_min, 1.1 * y_max, 350)

        xx, yy = np.meshgrid(x, y)
        xxyy = np.array([xx.ravel(), yy.ravel()]).T
        p = self.density(xxyy).reshape(xx.shape)
        fig, ax = plt.subplots(1, 1)
        ax.scatter(X[:, 0], X[:, 1], color='r', marker='x')
        min_pwr = int(np.min(np.log10(p)))
        levels = np.append(10**min_pwr, np.logspace(-3, 0, 15))
        cntr = ax.contourf(x, y, p, levels, alpha=0.75,
                           norm=colors.LogNorm(vmin=1e-3, vmax=1.))
        fig.colorbar(cntr, format='%.0e')

        plt.scatter(self.mu[:, 0], self.mu[:, 1], color='m', marker='o')
        ax.set_xlabel('$x_1$', fontsize=16)
        ax.set_ylabel('$x_2$', fontsize=16)
        ax.set_title('GMM Density Estimate, $Q = %0.4f$' % self.Q_best)
        return fig
