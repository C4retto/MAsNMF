from dataclasses import dataclass, field
import numpy as np
import networkx as nx
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.io as sio


class MAsNMF(object):
    '''
    模块化非对称非负矩阵分解算法类.
    '''
    def __init__(self, n_components, iterations, _lambda, random_state=42):
        self.n_components = n_components
        self.iterations = iterations
        self._lambda = _lambda
        self.random_state = random_state

    def init_matrices(self):
        rng = np.random.default_rng(seed=self.random_state)

        n = self.A.shape[0]
        self.W = rng.random((n, self.n_components))
        self.H = rng.random((self.n_components, self.n_components))

    def update_W(self):
        pass

    def update_H(self):
        pass

    def update_hatW(self):
        pass

    def fit(self, adjacency_matrix):
        self.A = adjacency_matrix

        k_out = self.A.sum(axis=1)   # (n,)
        k_in  = self.A.sum(axis=0)   # (n,)
        m = self.A.sum()
        self.B1 = np.outer(k_out, k_in) / m

        self.init_matrices()

        for _ in range(self.iterations):
            self.update_W()
            self.update_H()
            self.update_hatW()



if __name__ == "__main__":
    pass