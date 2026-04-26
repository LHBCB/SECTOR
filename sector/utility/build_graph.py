import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from .scale_rule import resolve_large_scale_mode


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


# ------------------------------------------------------------------
# Dense behavior for small datasets
# ------------------------------------------------------------------
def _knn_adjacency_matrix_dense(adata, k: int = 6, include_self: bool = False) -> np.ndarray:
    """
    Build a symmetric k-NN (by Euclidean distance) adjacency matrix (int64, {0,1}).
    This is the original dense implementation and is preserved exactly for small datasets.
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

    adj = (adj + adj.T) > 0
    return adj.astype(np.int64)


def _radius_adjacency_matrix_dense(adata, max_distance: float) -> np.ndarray:
    """
    Build a (strict) radius graph adjacency: A[i,j] = 1 if dist(i,j) < max_distance.
    This is the original dense implementation and is preserved exactly for small datasets.
    """
    assert 'spatial' in adata.obsm, 'AnnData object should have provided spatial information'
    coords = adata.obsm['spatial']
    dist = pairwise_distances(coords, metric='euclidean')
    adj = (dist < max_distance).astype(np.int64)
    return adj


def _build_spatial_graph_dense_mode(
    adata,
    n: int = 6,
    dmax: float = 50.0,
    mode: str = 'KNN',
    weight_mode: str = 'gaussian',
    sigma: float | None = None
):
    """
    Dense implementation, kept for small datasets.
    """
    assert 'spatial' in adata.obsm, 'AnnData object should have provided spatial information'
    coords = adata.obsm['spatial']
    dist_full = pairwise_distances(coords, metric='euclidean')

    if mode == 'KNN':
        adj_bool = _knn_adjacency_matrix_dense(adata, k=n, include_self=False)
    else:
        adj_bool = _radius_adjacency_matrix_dense(adata, max_distance=dmax)

    adj_bool = sp.coo_matrix(adj_bool)
    adj_bool.setdiag(0)
    adj_bool.eliminate_zeros()
    adj_bool = adj_bool.tocoo()

    rows, cols = adj_bool.row, adj_bool.col
    neighbor_dists = dist_full[rows, cols]

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
    adj_w_no_self = sp.coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))
    adj_norm = _normalize_adjacency_with_self_loops(adj_w_no_self.tocsr()).coalesce()

    adj_w_self = (adj_w_no_self + sp.eye(n_nodes, dtype=np.float32)).tocoo()
    indices = torch.from_numpy(
        np.vstack([adj_w_self.row, adj_w_self.col]).astype(np.int64)
    )
    values = torch.from_numpy(adj_w_self.data.astype(np.float32))
    adj_label = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()

    nnz_no_self = adj_w_no_self.nnz
    norm_value = (n_nodes * n_nodes) / float((n_nodes * n_nodes - nnz_no_self) * 2)

    return {
        'adj_norm': adj_norm,
        'adj_label': adj_label,
        'norm_value': norm_value,
    }


# ------------------------------------------------------------------
# Sparse behavior for large datasets
# ------------------------------------------------------------------
def _build_knn_edges_sparse(coords: np.ndarray, k: int):
    n = coords.shape[0]
    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        metric='euclidean',
        algorithm='kd_tree',
        n_jobs=-1,
    )
    nbrs.fit(coords)
    dists, inds = nbrs.kneighbors(coords, return_distance=True)

    dists = dists[:, 1:]
    inds = inds[:, 1:]

    rows = np.repeat(np.arange(n, dtype=np.int64), inds.shape[1])
    cols = inds.reshape(-1).astype(np.int64)
    dists = dists.reshape(-1).astype(np.float32)
    return rows, cols, dists


def _build_radius_edges_sparse(coords: np.ndarray, dmax: float):
    nbrs = NearestNeighbors(
        radius=dmax,
        metric='euclidean',
        algorithm='kd_tree',
        n_jobs=-1,
    )
    nbrs.fit(coords)
    d_list, i_list = nbrs.radius_neighbors(coords, sort_results=True)

    rows_all, cols_all, dists_all = [], [], []
    for r, (di, ii) in enumerate(zip(d_list, i_list)):
        mask = ii != r
        if np.any(mask):
            rows_all.append(np.full(mask.sum(), r, dtype=np.int64))
            cols_all.append(ii[mask].astype(np.int64))
            dists_all.append(di[mask].astype(np.float32))

    if not rows_all:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    dists = np.concatenate(dists_all)
    return rows, cols, dists


def _build_spatial_graph_sparse(
    adata,
    n: int = 6,
    dmax: float = 50.0,
    mode: str = 'KNN',
    weight_mode: str = 'gaussian',
    sigma: float | None = None
):
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float32)
    n_nodes = coords.shape[0]

    if mode == 'KNN':
        rows, cols, neighbor_dists = _build_knn_edges_sparse(coords, n)
    else:
        rows, cols, neighbor_dists = _build_radius_edges_sparse(coords, dmax)

    if sigma is None:
        positive = neighbor_dists[neighbor_dists > 0]
        sigma = float(np.median(positive)) if positive.size > 0 else 1.0

    if weight_mode == 'gaussian':
        weights = np.exp(-(neighbor_dists ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    elif weight_mode == 'inverse':
        weights = (1.0 / (neighbor_dists + 1e-12)).astype(np.float32)
        q95 = np.percentile(weights, 95) if weights.size > 0 else 1.0
        weights = np.clip(weights / (q95 + 1e-12), 0.0, 1.0).astype(np.float32)
    else:
        weights = np.ones_like(neighbor_dists, dtype=np.float32)

    A = sp.coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A = A.maximum(A.T).tocsr()

    adj_norm = _normalize_adjacency_with_self_loops(A)
    adj_label = _scipy_to_torch_sparse_tensor(
        A + sp.eye(n_nodes, dtype=np.float32, format='csr')
    ).coalesce()

    nnz_no_self = A.nnz
    norm_value = (n_nodes * n_nodes) / float((n_nodes * n_nodes - nnz_no_self) * 2)

    return {
        'adj_norm': adj_norm,
        'adj_label': adj_label,
        'norm_value': norm_value,
    }


def build_spatial_graph(
    adata,
    n: int = 6,
    dmax: float = 50.0,
    mode: str = 'KNN',
    weight_mode: str = 'gaussian',
    sigma: float | None = None,
    large_scale: bool | None = None,
    large_scale_mode: str = 'auto',
    large_scale_n_obs_threshold: int = 100000,
):
    """
    Build the spatial graph.

    - For small datasets: dense implementation.
    - For large datasets: sparse implementation.
    """
    if large_scale is None:
        large_scale = resolve_large_scale_mode(
            large_scale_mode,
            len(adata),
            large_scale_n_obs_threshold,
        )

    if large_scale:
        return _build_spatial_graph_sparse(
            adata=adata,
            n=n,
            dmax=dmax,
            mode=mode,
            weight_mode=weight_mode,
            sigma=sigma,
        )

    return _build_spatial_graph_dense_mode(
        adata=adata,
        n=n,
        dmax=dmax,
        mode=mode,
        weight_mode=weight_mode,
        sigma=sigma,
    )
