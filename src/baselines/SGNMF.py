import numpy as np

class SGNMF(object):
    '''
    SGNMF.
    @article{liu2023symmetry,
        title={Symmetry and graph bi-regularized non-negative matrix factorization for precise community detection},
        author={Liu, Zhigang and Luo, Xin and Zhou, Mengchu},
        journal={IEEE Transactions on Automation Science and Engineering},
        volume={21},
        number={2},
        pages={1406--1420},
        year={2023},
        publisher={IEEE}
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
        self.W = rng.random((n, self.n_components))
        self.Y = rng.random((n, self.n_components))     # A \approx W \otimes Y^\top
    
    def update_W(self):
        numer = self.A @ self.Y + self._alpha * self.Y
        denom = self.W @ (self.Y.T @ self.Y) + self._alpha * self.W
        denom = np.maximum(denom, 1e-12)
        self.W *= numer / denom

    def update_Y(self):
        numer = self.A.T @ self.W + self._alpha * self.W \
        + self._lambda * self.A @ self.Y
        denom = self.Y @ (self.W.T @ self.W) + self._alpha * self.Y \
        + self._lambda * self.D @ self.Y
        denom = np.maximum(denom, 1e-12)
        self.Y *= numer / denom

    def fit(self, adjacency_matrix):
        self.A = adjacency_matrix
        deg = np.asarray(self.A.sum(axis=1)).ravel()
        self.D = np.diag(deg)

        self.init_matrices()
        for _ in range(self.iterations):
            self.update_W()
            self.update_Y()

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

    model = SGNMF(n_components=n_clusters, iterations=100, _lambda=0.1, _alpha=0.1, random_state=42)
    model.fit(adjacency_matrix)
    pred_labels = np.argmax(model.W, axis=1)

    print("NMI:", NMI(true_labels, pred_labels))   # 0.0668
    print("ARI:", ARI(true_labels, pred_labels))   # 0.0901
    print("ACC:", ACC(true_labels, pred_labels))   # 0.1366