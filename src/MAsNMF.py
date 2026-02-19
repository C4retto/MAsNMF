import numpy as np


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
        self.hatW = rng.random((n, self.n_components))
        self.H = rng.random((self.n_components, self.n_components))

    def update_W(self):
        numer = self.X @ self.W @ self.H.T + self.X.T @ self.W @ self.H \
        + self._lambda * self.hatW

        denom = self.W @ self.H @ self.W.T @ self.W @ self.H.T \
        + self.W @ self.H.T @ self.W.T @ self.W @ self.H \
        + self._lambda
        denom = np.maximum(denom, 1e-12)

        self.W *= (numer / denom) ** 0.25

    def update_H(self):
        numer = self.W.T @ self.X @ self.W

        denom = self.W.T @ self.W @ self.H @ self.W.T @ self.W
        denom = np.maximum(denom, 1e-12)

        self.H *= numer / denom

    def update_hatW(self):
        numer = self.A @ self.hatW + self.A.T @ self.hatW \
        + 2 * self._lambda * self.W

        denom = self.B1 @ self.hatW + self.B1.T @ self.hatW \
        + 2 * self._lambda * self.hatW
        denom = np.maximum(denom, 1e-12)

        self.hatW *= numer / denom

        row_sums = self.hatW.sum(axis=1, keepdims=True)
        self.hatW /= np.maximum(row_sums, 1e-12)

    def fit(self, adjacency_matrix, _alpha=1):
        self.A = adjacency_matrix

        k_out = self.A.sum(axis=1)   # (n,)
        k_in  = self.A.sum(axis=0)   # (n,)
        m = self.A.sum()
        self.B1 = np.outer(k_out, k_in) / m

        self.X = _alpha * self.A

        self.init_matrices()

        for _ in range(self.iterations):
            self.update_W()
            self.update_H()
            self.update_hatW()



if __name__ == "__main__":
    # 示例用法, texas 数据集
    from torch_geometric.datasets import WebKB
    from torch_geometric.utils import to_networkx
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI, accuracy_score as ACC

    data = WebKB(root="./data/WebKB", name="texas")[0]
    graph = to_networkx(data)
    adjacency_matrix = nx.adjacency_matrix(graph).toarray()
    n_clusters = len(np.unique(data.y))
    true_labels = data.y

    model = MAsNMF(n_components=n_clusters, iterations=100, _lambda=0.1, random_state=42)
    model.fit(adjacency_matrix, _alpha=1)
    pred_labels = np.argmax(model.W, axis=1)

    print("NMI:", NMI(true_labels, pred_labels))    # 0.2258
    print("ARI:", ARI(true_labels, pred_labels))    # 0.2072
    print("ACC:", ACC(true_labels, pred_labels))    # 0.2350