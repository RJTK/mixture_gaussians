"""
Implements EM algorithm for fitting a mixture of gaussian densities.
"""
import logging

import numpy as np
import matplotlib.colors as colors

from matplotlib import pyplot as plt

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
    def __init__(self, num_clusters, do_logging=False, prune_clusters=True):
        self.num_clusters = num_clusters
        self._fitted = False
        # If a cluster has near 0 weight or a near singular variance
        # we will prune this cluster and reduce the number of clusters
        # by 1.  This should never occur if num_clusters=1 except in
        # extremely degenerate cases
        self.prune_clusters = prune_clusters
        self.logging = logging
        if self.logging:
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
            if self.logging:
                self.logger.error('LinAlgError in Estep: %s' % e)
            raise EstepException('LinAlgError')

        M = np.max(log_Px, axis=0)
        logsum_piPx = M + np.log(np.einsum('k,ki->i', pi,
                                           np.exp(log_Px - M)))
        r = (pi[:, None] * np.exp(log_Px)) / np.exp(logsum_piPx)
        if np.any(np.isnan(r)):
            if self.logging:
                self.logger.error('r contains nan r = %s, pi = %s\n'
                                  'Setting nans to 0 and continuing' % (r, pi))
            r[np.isnan(r)] = 0

        return r

    def _Mstep(self, X, r):
        # @numba.jit(nopython=True, cache=True)
        N = X.shape[0]
        pi = (1. / N) * np.sum(r, axis=1)
        mu = np.einsum('ki,ip->kp', r, X) / (N * pi[:, None])
        Sigma = (np.einsum('ki,ip,iq->kpq',
                           r, X, X) / (N * pi[:, None, None]) -
                 np.einsum('kp,kq->kpq', mu, mu))
        return pi, mu, Sigma

    def _Qfunc(self, X, r, pi, mu, Sigma):
        log_pi = np.log(pi)
        if np.any(np.isnan(log_pi)):
            if self.logging:
                self.logger.error('nan in log_pi.  pi = %s' % pi)
            raise QstepException('nan in log_pi')

        cluster_quality = np.sum(log_pi @ r)
        U = np.moveaxis(X[:, None, :] - mu, 0, 1)

        try:
            nll = -0.5 * (np.einsum('ki,k->', r, np.linalg.slogdet(Sigma)[1])
                          + np.einsum('ki,kip,kpq,kiq->',
                                      r, U, np.linalg.inv(Sigma), U))
        except np.linalg.LinAlgError as e:
            if self.logging:
                self.logger.error('LinAlgError in Q: %s' % e)
            raise QstepException('LinAlgError')

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
                if self.logging:
                    self.logger.warning('Pruned a cluster!')
            else:
                k += 1
        return r, pi, mu, Sigma

    def fit(self, X, num_restarts=1, max_steps=50,
            eps=1e-6, debug=False, callback=None):
        N, p = X.shape
        K = self.num_clusters

        for _ in range(num_restarts):
            if self.logging:
                self.logger.info('----- EM restart ------')

            # Random initialization
            mu = np.random.normal(size=(K, p))
            Sigma = np.stack([np.eye(p)] * K)
            pi = np.abs(np.random.normal(size=K))
            pi = pi / np.sum(pi)

            pi_best, mu_best, Sigma_best = pi, mu, Sigma

            # Intialization for loop conditions
            Q_best = -np.inf
            Q_prev = -np.inf
            delta_Q = np.inf
            step_count = 0

            while delta_Q > eps and step_count < max_steps:
                step_count += 1
                try:
                    r = self._Estep(X, pi, mu, Sigma)
                except EstepException as e:
                    if self.logging:
                        self.logger.error('Caught EstepException %s.  '
                                          'Continuing on next random restart'
                                          % e)
                    break

                try:
                    pi, mu, Sigma = self._Mstep(X, r)
                except MstepException as e:
                    if self.logging:
                        logging.error('Caught MstepException %s.  '
                                      'Continuing on next random restart' % e)
                    break

                if self.prune_clusters:
                    r, pi, mu, Sigma = self._prune(r, pi, mu, Sigma)

                try:
                    Q = self._Qfunc(X, r, pi, mu, Sigma)
                except QstepException as e:
                    if self.logging:
                        self.logger.error('Caught QstepException %s.  '
                                          'Continuing on next random restart'
                                          % e)
                    break

                delta_Q = Q - Q_prev
                Q_prev = Q
                if self.logging:
                    self.logger.debug("Step %d: Q = %0.5f, delta_Q = %0.5f"
                                      % (step_count, Q, delta_Q))
                    if delta_Q < -np.abs(eps):
                        self.logger.error("Q did not decrease!")

            if Q > Q_best:
                Q_best = Q
                pi_best, mu_best, Sigma_best = pi, mu, Sigma

            if callback is not None:
                callback(X, pi_best, mu_best, Sigma_best, Q_best)

            if self.logging:
                self.logger.debug('Done.  pi_best = %s, mu_best = %s, '
                                  'Sigma_best = %s'
                                  % (pi_best, mu_best, Sigma_best))

        if callback is not None:
            callback(X, pi_best, mu_best, Sigma_best, Q_best)

        self.pi, self.mu, self.Sigma = pi_best, mu_best, Sigma_best
        if self.prune_clusters:
            K = self.mu.shape[0]
            if K < self.num_clusters:
                self.num_clusters = K
                if self.logging:
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
            if self.logging:
                self.logger.error('density p contains nan p = %s, log_p = %s\n'
                                  'Setting nans to 0 and continuing'
                                  % (p, log_Px))
            p[np.isnan(p)] = 0
        return p

    def plot_scatter_density(self, X, fit=True):
        """Plots the data X and the density estimate of the GMM"""
        if not self._fitted:
            if fit:
                self.fit(X)
            else:
                raise AssertionError('Model not fit and specified fit=False')

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
        levels = np.append(10**min_pwr, np.logspace(-2, 0, 15))
        cntr = ax.contourf(x, y, p, levels, alpha=0.75,
                           norm=colors.LogNorm(vmin=1e-3, vmax=1.))
        fig.colorbar(cntr, format='%.0e')

        plt.scatter(self.mu[:, 0], self.mu[:, 1], color='m', marker='o')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('GMM Density Estimate')
        return fig
