import numpy as np

class MtriNMF(object):
    '''
    MtriNMF.
    @article{yan2019modularized,
        title={Modularized tri-factor nonnegative matrix factorization for community detection enhancement},
        author={Yan, Chao and Chang, Zhenhai},
        journal={Physica A: Statistical Mechanics and its Applications},
        volume={533},
        pages={122050},
        year={2019},
        publisher={Elsevier}
        }
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
        numer = self.A @ self.W @ self.H.T + self.A.T @ self.W @ self.H \
        + self._lambda * self.A.T @ self.W

        denom = self.W @ self.H @ self.W.T @ self.W @ self.H.T \
        + self.W @ self.H.T @ self.W.T @ self.W @ self.H \
        + self._lambda * self.B1.T @ self.W
        denom = np.maximum(denom, 1e-12)

        self.W *= (numer / denom) ** 0.25

    def update_H(self):
        numer = self.W.T @ self.A @ self.W

        denom = self.W.T @ self.W @ self.H @ self.W.T @ self.W
        denom = np.maximum(denom, 1e-12)

        self.H *= numer / denom

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


if __name__ == "__main__":
    from torch_geometric.datasets import WebKB
    from torch_geometric.utils import to_networkx
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI

    data = WebKB(root="./data/WebKB", name="texas")[0]
    graph = to_networkx(data)
    adjacency_matrix = nx.adjacency_matrix(graph).toarray()
    n_clusters = len(np.unique(data.y))
    true_labels = data.y

    model = MtriNMF(n_components=n_clusters, iterations=100, _lambda=0.1, random_state=42)
    model.fit(adjacency_matrix)
    pred_labels = np.argmax(model.W, axis=1)

    print("NMI:", NMI(true_labels, pred_labels))   # 0.1867
    print("ARI:", ARI(true_labels, pred_labels))   # 0.1639