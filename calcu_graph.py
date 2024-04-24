import numpy as np
import torch
import scipy.sparse as sp


def construct_graph(features, k):
    dist = np.zeros((features.shape[0], features.shape[0]))
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            if i != j:
                dist[i][j] = np.exp(-np.sum(np.square(features[i] - features[j])) * 0.2)

    topk = int(k)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    adj = np.zeros_like(dist)
    for i, v in enumerate(inds):
        for k in v:
            adj[i][k] = 1

    return adj


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.dot(r_mat_inv)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def squared_distance(X, Y=None):
    '''
    Calculates the pairwise distance between points in X and Y

    X:          n x d matrix
    Y:          m x d matrix
    W:          affinity -- if provided, we normalize the distance

    returns:    n x m matrix of all pairwise squared Euclidean distances
    '''
    if Y is None:
        Y = X
    X1 = torch.reshape(X, (1, X.shape[0], X.shape[-1]))
    Y1 = torch.reshape(Y, (Y.shape[0], 1, Y.shape[-1]))
    DXY = torch.sum(torch.square(X1 - Y1), dim=-1)

    return DXY
