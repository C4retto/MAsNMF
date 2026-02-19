import numpy as np

class RoAsNMF(object):
    '''
    Rewrite in Python. 
    本文考虑网络中离群边对社区检测的影响, 提出一种鲁棒的非对称非负矩阵分解方法, 以提高社区检测的准确性和稳定性.
    
    @article{li2025robust,
        title={Robust asymmetric nonnegative matrix factorization for community detection in directed network},
        author={Li, Kangmin and Yu, Yongguang and Hu, Wei and Cui, Yibing and Li, Jiting},
        journal={Chaos, Solitons \& Fractals},
        volume={200},
        pages={117144},
        year={2025},
        publisher={Elsevier}
        }
    '''

    def __init__(self, n_components, iterations, _lambda, _alpha, random_state=42):
        self.n_components = n_components
        self.iterations = iterations
        self._lambda = _lambda
        self._alpha = _alpha
        self.random_state = random_state

    def init_matrices(self):
        rng = np.random.default_rng(seed=self.random_state)

        n = self.A.shape[0]
        self.U = rng.random((n, n))
        self.W = rng.random((n, self.n_components))
        self.H = rng.random((self.n_components, self.n_components))

    def update_U(self):
        Q = np.diag(1 / np.maximum(np.sqrt(((self.A - self.U) ** 2).sum(axis=1)), 1e-12))

        numer = self._alpha * self.A @ Q \
        + self.W @ self.H @ self.W.T

        denom = self._alpha * self.U @ Q \
        + self.U
        denom = np.maximum(denom, 1e-12)

        self.U *= (numer / denom) ** 0.5

    def update_W(self):
        numer = self.U @ self.W @ self.H.T + self.U.T @ self.W @ self.H \
        + self._lambda

        denom = self.W @ self.H @ self.W.T @ self.W @ self.H.T \
        + self.W @ self.H.T @ self.W.T @ self.W @ self.H \
        + self._lambda * np.repeat(self.W.sum(axis=1, keepdims=True), self.n_components, axis=1)
        denom = np.maximum(denom, 1e-12)

        self.W *= (numer / denom) ** 0.25

    def update_H(self):
        numer = self.W.T @ self.U @ self.W

        denom = self.W.T @ self.W @ self.H @ self.W.T @ self.W
        denom = np.maximum(denom, 1e-12)

        self.H *= numer / denom

    def fit(self, adjacency_matrix):
        self.A = adjacency_matrix

        self.init_matrices()
        for _ in range(self.iterations):
            self.update_U()
            self.update_W()
            self.update_H()

if __name__ == "__main__":    # 示例用法, texas 数据集
    from torch_geometric.datasets import WebKB
    from torch_geometric.utils import to_networkx
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI

    data = WebKB(root="./data/WebKB", name="texas")[0]
    graph = to_networkx(data)
    adjacency_matrix = nx.adjacency_matrix(graph).toarray()
    true_labels = data.y.numpy()

    model = RoAsNMF(n_components=6, iterations=100, _lambda=0.1, _alpha=0.5)
    model.fit(adjacency_matrix)
    pred_labels = np.argmax(model.W, axis=1)    

    print("NMI:", NMI(true_labels, pred_labels))   # 0.1645
    print("ARI:", ARI(true_labels, pred_labels))   # 0.0543