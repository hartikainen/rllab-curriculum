from __future__ import print_function
from __future__ import absolute_import
from rllab.distributions.base import Distribution
import theano.tensor as TT
import numpy as np


class Bernoulli(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_p = old_dist_info_vars["p"]
        new_p = new_dist_info_vars["p"]
        kl = old_p * (TT.log(old_p + 1e-8) - TT.log(new_p + 1e-8)) + \
             (1 - old_p) * (TT.log(1 - old_p + 1e-8) - TT.log(1 - new_p + 1e-8))
        return TT.sum(kl, axis=-1)

    def kl(self, old_dist_info, new_dist_info):
        old_p = old_dist_info["p"]
        new_p = new_dist_info["p"]
        kl = old_p * (np.log(old_p + 1e-8) - np.log(new_p + 1e-8)) + \
             (1 - old_p) * (np.log(1 - old_p + 1e-8) - np.log(1 - new_p + 1e-8))
        return np.sum(kl, axis=-1)

    def sample(self, dist_info):
        p = np.asarray(dist_info["p"])
        return np.cast['int'](np.random.uniform(low=0., high=1., size=p.shape) < p)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        old_p = old_dist_info_vars["p"]
        new_p = new_dist_info_vars["p"]
        return TT.prod(x_var * new_p / (old_p + 1e-8) + (1 - x_var) * (1 - new_p) / (1 - old_p + 1e-8), axis=-1)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        p = dist_info_vars["p"]
        return TT.sum(x_var * TT.log(p + 1e-8) + (1 - x_var) * TT.log(1 - p + 1e-8), axis=-1)

    def log_likelihood(self, xs, dist_info):
        p = dist_info["p"]
        return np.sum(xs * np.log(p + 1e-8) + (1 - xs) * np.log(1 - p + 1e-8), axis=-1)

    def entropy(self, dist_info):
        p = dist_info["p"]
        return np.sum(- p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8), axis=-1)

    @property
    def dist_info_keys(self):
        return ["p"]