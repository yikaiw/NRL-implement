import scipy.sparse as sp
import numpy as np
import time


'''
This dataset consists of 169,209 users and 500,000 "following" relationship. We divide it into 6 files.
- adj_train.npz: sparse adjacency matrix of graph. The weight associated with each directed edge is 1.0. You can use it for training and load graph via load_npz function in scipy.sparse package.
- train_edges.npy: training edges in graph, in #n_train * 2 numpy array. Each row represents two node id with directed edges. You can load edges via load function in numpy. The following files have the same format as this file.
- val_edges.npy: positive validation edges in graph, in #n_valid * 2 numpy array. We select 5% edges for validation.
- val_edges_false.npy: negative validation edges in graph, in #n_valid_neg * 2 numpy array. The negative edges do not exist in graph.
- test_edges.npy: positive test edges in graph, in #n_test * 2 numpy array. We select 10% edges for test.
- test_edges_false.npy: negative test edges in graph, in #n_test_neg * 2 numpy array. The negative edges do not exist in graph.
'''

def load_tencent(path="data/tencent/", dataset="tencent"):
    adj_train = sp.load_npz(path + "adj_train.npz")
    adj_train = adj_train + adj_train.T.multiply(adj_train.T > adj_train) - adj_train.multiply(adj_train.T > adj_train)
    return None, adj_train, None


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if dataset == "tencent":
        return load_tencent(path=path, dataset=dataset)

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels


def get_splits(typ, y=None):
    if typ == "cora":
        idx_train = range(200)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
        y_train = np.zeros(y.shape, dtype=np.int32)
        y_val = np.zeros(y.shape, dtype=np.int32)
        y_test = np.zeros(y.shape, dtype=np.int32)
        y_train[idx_train] = y[idx_train]
        y_val[idx_val] = y[idx_val]
        y_test[idx_test] = y[idx_test]
        return y_train, y_val, y_test, idx_train, idx_val, idx_test
    else:   # typ == "cora":
        # adj_train = sp.load_npz(y + "adj_train.npz")
        # train_edges = np.load(y + "train_edges.npy")
        test_edges = np.load(y + "test_edges.npy")   # select 10% edges
        test_edges_false = np.load(y + "test_edges_false.npy")
        return test_edges, test_edges_false


if __name__ == '__main__':
    # usage
    X, A, y = load_data(path="data/cora/", dataset='cora')
    # y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y)

    # _, A, _ = load_tencent()
    # get_splits(typ="tencent", y="data/tencent/")
    print(X)
