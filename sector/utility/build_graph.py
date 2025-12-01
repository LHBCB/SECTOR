import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances

def _scipy_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a PyTorch sparse COO tensor (float32)."""
    coo = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64)
    )
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, size=shape, dtype=values.dtype)


def _normalize_adjacency_with_self_loops(adj_no_self: sp.spmatrix) -> torch.Tensor:
    """
    Symmetric normalize (A + I) with D^{-1/2} (A + I) D^{-1/2}, returning a torch sparse tensor.
    Expects a square SciPy sparse matrix without self-loops.
    """
    a_hat = adj_no_self + sp.eye(adj_no_self.shape[0], dtype=adj_no_self.dtype)
    row_sum = np.array(a_hat.sum(1))
    d_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    a_norm = a_hat.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt).tocoo()
    return _scipy_to_torch_sparse_tensor(a_norm).coalesce()


def _knn_adjacency_matrix(adata, k: int = 6, include_self: bool = False) -> np.ndarray:
    """
    Build a symmetric k-NN (by Euclidean distance) adjacency matrix (int64, {0,1}).
    Behavior mirrors the original:
      - for each node, take the k nearest *others* + itself (k+1),
      - optionally drop self,
      - symmetrize with A = A + A^T, then binarize.
    """
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'
    coords = adata.obsm['spatial']
    dist = pairwise_distances(coords)

    n_obs = len(adata)
    adj = np.zeros((n_obs, n_obs), dtype=np.int64)

    for i in range(n_obs):
        nn = np.argsort(dist[i, :])[:k + 1]  # includes self
        adj[i, nn] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj)
        adj[x, y] = 0

    # make symmetric and binary
    adj = (adj + adj.T) > 0
    return adj.astype(np.int64)


def _radius_adjacency_matrix(adata, max_distance: float) -> np.ndarray:
    """
    Build a (strict) radius graph adjacency: A[i,j] = 1 if dist(i,j) < max_distance.
    Returns int64 matrix with {0,1}.
    """
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'
    coords = adata.obsm['spatial']
    dist = pairwise_distances(coords, metric='euclidean')
    adj = (dist < max_distance).astype(np.int64)
    return adj

def build_spatial_graph(
    adata,
    n: int = 6,
    dmax: float = 50.0,
    mode: str = 'KNN',
    weight_mode: str = 'gaussian',
    sigma: float | None = None
):
    """
    Construct a weighted spatial graph from AnnData coordinates (adata.obsm['spatial']).

    Parameters
    ----------
    adata : AnnData
        Must contain 'spatial' in `adata.obsm` (N x 2 or N x d array of coordinates).
    n : int, default 6
        k for k-NN if mode == 'KNN'.
    dmax : float, default 50.0
        Distance threshold if mode != 'KNN'.
    mode : {'KNN', other}, default 'KNN'
        If 'KNN', build k-NN graph; else build radius graph with threshold `dmax`.
    weight_mode : {'gaussian', 'inverse', 'binary'}, default 'gaussian'
        How to transform neighbor distances into edge weights.
    sigma : float or None, default None
        Bandwidth for Gaussian weights; if None, use median neighbor distance (>0).

    Returns
    -------
    dict with keys:
        - 'adj_norm'  : torch.sparse_coo_tensor (normalized (A+I))
        - 'adj_label' : torch.sparse_coo_tensor (A with self-loops, weights)
        - 'norm_value': float (kept as in original implementation)
    """
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'
    coords = adata.obsm['spatial']
    dist_full = pairwise_distances(coords, metric='euclidean')

    # 1) Boolean adjacency (no self-loops)
    if mode == 'KNN':
        adj_bool = _knn_adjacency_matrix(adata, k=n, include_self=False)
    else:
        adj_bool = _radius_adjacency_matrix(adata, max_distance=dmax)

    adj_bool = sp.coo_matrix(adj_bool)
    adj_bool.setdiag(0)
    adj_bool.eliminate_zeros()
    adj_bool = adj_bool.tocoo()

    rows, cols = adj_bool.row, adj_bool.col
    neighbor_dists = dist_full[rows, cols]

    # 2) Edge weights derived from distances
    if sigma is None:
        positive = neighbor_dists[neighbor_dists > 0]
        sigma = (np.median(positive) if positive.size > 0 else 1.0) + 1e-12

    if weight_mode == 'gaussian':
        weights = np.exp(- (neighbor_dists ** 2) / (2.0 * (sigma ** 2))).astype(np.float32)
    elif weight_mode == 'inverse':
        weights = (1.0 / (neighbor_dists + 1e-12)).astype(np.float32)
        q95 = np.percentile(weights, 95) if weights.size > 0 else 1.0
        weights = np.clip(weights / (q95 + 1e-12), 0.0, 1.0).astype(np.float32)
    else:  # 'binary'
        weights = np.ones_like(neighbor_dists, dtype=np.float32)

    n_nodes = adj_bool.shape[0]

    # Weighted adjacency without self-loops (SciPy)
    adj_w_no_self = sp.coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

    # Normalized adjacency (adds self-loops inside)
    adj_norm = _normalize_adjacency_with_self_loops(adj_w_no_self.tocsr()).coalesce()

    # Weighted adjacency with self-loops (weight 1.0 on the diagonal) for labels/edge_index
    adj_w_self = (adj_w_no_self + sp.eye(n_nodes, dtype=np.float32)).tocoo()
    indices = torch.from_numpy(
        np.vstack([adj_w_self.row, adj_w_self.col]).astype(np.int64)
    )
    values = torch.from_numpy(adj_w_self.data.astype(np.float32))
    adj_label = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()

    # Keep the original placeholder 'norm_value' logic intact
    nnz_no_self = adj_w_no_self.nnz
    norm_value = (n_nodes * n_nodes) / float((n_nodes * n_nodes - nnz_no_self) * 2)

    return {
        "adj_norm": adj_norm,
        "adj_label": adj_label,
        "norm_value": norm_value,
    }
