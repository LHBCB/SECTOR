import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.data import Data

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import re
import scanpy as sc

from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
import scipy.sparse
from scipy.optimize import linear_sum_assignment

from .build_graph import build_spatial_graph
from .scale_rule import resolve_large_scale_mode

def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {'1', 'true', 't', 'yes', 'y'}
    return bool(x)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------
# SVG functions
# ------------------------------------------------------------------
def _build_svg_knn_graph(coords, k=6):
    coords = np.asarray(coords, dtype=np.float64)
    n_obs = coords.shape[0]
    if n_obs == 0:
        return scipy.sparse.csr_matrix((0, 0), dtype=np.float64)
    if n_obs == 1:
        return scipy.sparse.csr_matrix((1, 1), dtype=np.float64)

    k_eff = min(max(1, int(k)), n_obs - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean', algorithm='auto')
    nbrs.fit(coords)
    _, idx = nbrs.kneighbors(coords, return_distance=True)

    rows = np.repeat(np.arange(n_obs, dtype=np.int64), k_eff)
    cols = idx[:, 1:].reshape(-1).astype(np.int64)
    vals = np.ones(rows.shape[0], dtype=np.float64)

    W = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs), dtype=np.float64).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    W = W.maximum(W.T).tocsr().astype(np.float64)
    return W


def _compute_moran_scores(X, W):
    n_obs = X.shape[0]
    if n_obs == 0 or X.shape[1] == 0:
        return np.array([], dtype=np.float64)

    S0 = float(W.sum())
    if S0 <= 0:
        return np.full(X.shape[1], -np.inf, dtype=np.float64)

    deg = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)

    if scipy.sparse.issparse(X):
        X = X.tocsr().astype(np.float64)
        sums = np.asarray(X.sum(axis=0)).ravel()
        sq_sums = np.asarray(X.multiply(X).sum(axis=0)).ravel()
        WX = W.dot(X)
        xTwx = np.asarray(X.multiply(WX).sum(axis=0)).ravel()
        dTx = np.asarray(X.T.dot(deg)).ravel()
    else:
        X = np.asarray(X, dtype=np.float64)
        sums = X.sum(axis=0)
        sq_sums = np.square(X).sum(axis=0)
        WX = W.dot(X)
        xTwx = np.multiply(X, WX).sum(axis=0)
        dTx = deg @ X

    means = sums / float(n_obs)
    num = xTwx - 2.0 * means * dTx + (means ** 2) * S0
    den = sq_sums - (sums ** 2) / float(n_obs)

    scores = np.full_like(den, -np.inf, dtype=np.float64)
    valid = den > 1e-12
    scores[valid] = (float(n_obs) / S0) * (num[valid] / den[valid])
    scores[~np.isfinite(scores)] = -np.inf
    return scores


def _select_spatially_variable_genes(adata, n_top_genes=2000, spatial_k=6, flag_key='spatially_variable', score_key='svg_score'):
    if 'spatial' not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found; cannot compute SVGs.")

    n_genes = int(adata.n_vars)
    n_select = min(max(1, int(n_top_genes)), n_genes)

    W = _build_svg_knn_graph(adata.obsm['spatial'], k=spatial_k)
    scores = _compute_moran_scores(adata.X, W)

    order = np.argsort(-scores, kind='mergesort')
    selected = np.zeros(n_genes, dtype=bool)
    selected[order[:n_select]] = True

    adata.var[score_key] = scores
    adata.var[flag_key] = selected
    adata.var['highly_variable'] = selected
    return adata

    
# ------------------------------------------------------------------
# Small-data preprocessing: EXACT original behavior
# ------------------------------------------------------------------
def _adata_preprocess_small_dense(adata, min_cells=5, min_counts=10, 
                                  target_sum=1e4, n_comps=20, n_top_genes=2000,
                                  use_svg=False, spatial_k=6,
                                  random_seed=42):
    
    adata.layers['counts'] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    #sc.pp.log1p(adata)

    if _as_bool(use_svg):
        adata = _select_spatially_variable_genes(adata, n_top_genes=n_top_genes, spatial_k=spatial_k)
    else:
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', layer='counts', n_top_genes=n_top_genes)
    
    #adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=n_comps, svd_solver='auto', zero_center=True, random_state=random_seed)
    return adata

# ------------------------------------------------------------------
# Large-data preprocessing: current patch behavior
# ------------------------------------------------------------------

def _adata_preprocess_large_sparse(
    adata, min_cells=5, min_counts=10, target_sum=1e4, n_comps=20,
    n_top_genes=2000, random_seed=42, use_hvg_only=True,
    scale_before_pca=False, pca_zero_center=False, use_svg=False, spatial_k=6
):

    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    adata.layers['counts'] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # 1. CRITICAL: Log1p to stabilize variances.
    sc.pp.log1p(adata)

    # 2. Skip HVG selection if it's a small targeted panel (like Xenium)
    if adata.n_vars > n_top_genes:
        if _as_bool(use_svg):
            adata = _select_spatially_variable_genes(adata, n_top_genes=n_top_genes, spatial_k=spatial_k)
        else:
            sc.pp.highly_variable_genes(
                adata, flavor='seurat_v3', layer='counts', n_top_genes=n_top_genes,
            )
        if use_hvg_only and 'highly_variable' in adata.var:
            adata = adata[:, adata.var['highly_variable']].copy()
    elif _as_bool(use_svg):
        adata = _select_spatially_variable_genes(adata, n_top_genes=adata.n_vars, spatial_k=spatial_k)

    is_sparse = scipy.sparse.issparse(adata.X)
    
    # 3. CRITICAL: Densify the matrix if genes are low. 
    # 500k cells * 500 genes is ~1GB RAM. This allows true zero-centered scaling!
    if is_sparse and adata.n_vars <= 5000:
        adata.X = adata.X.toarray()
        is_sparse = False

    # Matrix is dense, apply standard scaling and centering
    if scale_before_pca or not is_sparse:
        sc.pp.scale(adata, max_value=10, zero_center=(False if is_sparse else True))
        #sc.pp.scale(adata)

    sc.pp.pca(
        adata,
        n_comps=min(n_comps, adata.n_vars - 1),
        svd_solver='arpack' if is_sparse else 'auto',
        zero_center=True if not is_sparse else bool(pca_zero_center),
        random_state=random_seed,
    )
    return adata

def read_st_data(args):
    adata = sc.read_h5ad(f'{args.dataset_path}/{args.dataset}/{args.slice}.h5ad')

    eval_mode = int(getattr(args, 'eval_mode', 1)) == 1

    if eval_mode:
        if args.label not in adata.obs:
            raise ValueError(f"Label column '{args.label}' not found in adata.obs.")
        adata = adata[~adata.obs[args.label].isna()].copy()

    large_scale = resolve_large_scale_mode(
        getattr(args, 'large_scale_mode', 'auto'),
        adata.n_obs,
        int(getattr(args, 'large_scale_n_obs_threshold', 100000)),
    )

    feature_mode = "SVG" if bool(getattr(args, "use_svg", False)) else "HVG"
    scale_mode = "large/sparse" if large_scale else "small/dense"
    print(f"[Preprocess] Using {feature_mode} features "
          f"(n_top_genes={args.n_top_genes}, mode={scale_mode})")
    
    graph_dict = build_spatial_graph(
        adata,
        n=args.k_s,
        weight_mode=args.weight_mode,
        large_scale=large_scale,
        large_scale_mode=getattr(args, 'large_scale_mode', 'auto'),
        large_scale_n_obs_threshold=int(getattr(args, 'large_scale_n_obs_threshold', 100000)),
    )
    edge_index = graph_dict['adj_label'].indices()
    edge_weight = graph_dict['adj_label'].values().float()

    if large_scale:
        adata = _adata_preprocess_large_sparse(
            adata,
            n_comps=args.n_comps,
            n_top_genes=args.n_top_genes,
            random_seed=args.seed,
            use_hvg_only=_as_bool(getattr(args, 'use_hvg_only', 1)),
            use_svg=_as_bool(getattr(args, 'use_svg', False)),
            spatial_k=args.k_s,
            scale_before_pca=_as_bool(getattr(args, 'scale_before_pca', 0)),
            pca_zero_center=_as_bool(getattr(args, 'pca_zero_center', 0)),
        )
        x = torch.from_numpy(np.asarray(adata.obsm['X_pca'], dtype=np.float32)).float()
    else:
        adata = _adata_preprocess_small_dense(
            adata,
            n_comps=args.n_comps,
            n_top_genes=args.n_top_genes,
            random_seed=args.seed,
            use_svg=_as_bool(getattr(args, 'use_svg', False)),
            spatial_k=args.k_s,
        )
        x = torch.from_numpy(adata.obsm['X_pca']).float()

    if eval_mode:
        labels = adata.obs[args.label].values
        categories = getattr(labels, 'categories', np.unique(labels))
        label_to_index = {label: idx for idx, label in enumerate(categories)}
        index_list = [label_to_index[lbl] for lbl in labels]
        y = torch.tensor(index_list, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    else:
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    return data, adata


def g_from_torchsparse(adj):
    adj = adj.coalesce()
    edge_index = adj.indices()
    edge_weight = adj.values()
    num_nodes = adj.size(0)
    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
    return data.to(edge_weight.device)


def index2adjacency(N, edge_index, weight=None, is_sparse=True):
    if is_sparse:
        m = edge_index.shape[1]
        weight = weight if weight is not None else torch.ones(m).to(edge_index.device)
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=weight, size=(N, N))
    else:
        adjacency = torch.zeros(N, N).to(edge_index.device)
        if weight is None:
            adjacency[edge_index[0], edge_index[1]] = 1
        else:
            adjacency[edge_index[0], edge_index[1]] = weight.reshape(-1)
    return adjacency


def adjacency2index(adjacency, weight=False):
    adj = adjacency
    edge_index = torch.nonzero(adj).t().contiguous()
    if weight:
        weight = adjacency[edge_index[0], edge_index[1]].reshape(-1)
        return edge_index, weight
    else:
        return edge_index


def select_activation(activation):
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation is None:
        return None
    else:
        raise NotImplementedError('the non_linear_function is not implemented')


def decoding_from_assignment(assignmatrix):
    pred = assignmatrix.argmax(dim=1)
    return pred

def remove_small_islands_on_graph(
    adj_sparse_torch,
    labels_np,
    min_abs=40,          # absolute island size threshold (tune per dataset)
    min_frac=0.1,       # relative to the largest component of that label
    max_iter=2           # run twice in case reassignment creates tiny remnants
):
    """
    Enforce at most one connected component per label on the spatial graph.
    Smaller components are reassigned to the neighbor label with the largest
    boundary weight (sum of edge weights crossing the boundary).
    """
    # torch.sparse_coo_tensor -> scipy.csr_matrix without self-loops
    A = adj_sparse_torch.coalesce()
    i = A.indices()[0].cpu().numpy()
    j = A.indices()[1].cpu().numpy()
    w = A.values().cpu().numpy()
    N = A.size(0)
    G = scipy.sparse.coo_matrix((w, (i, j)), shape=(N, N)).tocsr()
    G.setdiag(0); G.eliminate_zeros()

    y = labels_np.copy().astype(int)
    for _ in range(max_iter):
        changed = 0
        for c in np.unique(y):
            mask = (y == c)
            if mask.sum() == 0:
                continue

            # connected components in the label-induced subgraph
            Gc = G[mask][:, mask]
            n_comp, comp = connected_components(Gc, directed=False, return_labels=True)
            if n_comp <= 1:
                continue

            comp_sizes = np.bincount(comp)
            keep_size = comp_sizes.max()
            # islands are small components by absolute OR fractional size
            is_island = np.isin(comp, np.where(
                (comp_sizes < max(min_abs, int(min_frac * keep_size)))
            )[0])
            if not is_island.any():
                continue

            # original node indices of island nodes
            nodes_all = np.where(mask)[0]
            nodes_island = nodes_all[is_island]

            # for each island node: reassign to neighbor label with max boundary weight
            rows = G[nodes_island]  # list of CSR rows
            for r_idx, node in enumerate(nodes_island):
                nbr_idx = rows[r_idx].indices
                nbr_w   = rows[r_idx].data
                if nbr_idx.size == 0:
                    continue
                nbr_labels = y[nbr_idx]
                # accumulate weight per *different* label
                best_lb, best_w = c, 0.0
                for lb, ww in zip(nbr_labels, nbr_w):
                    if lb == c:   # same label, skip
                        continue
                if nbr_idx.size:
                    uniq, inv = np.unique(nbr_labels, return_inverse=True)
                    sums = np.bincount(inv, weights=nbr_w)
                    # exclude the current label if present
                    for u, s in zip(uniq, sums):
                        if u == c: 
                            continue
                        if s > best_w:
                            best_lb, best_w = int(u), float(s)

                if best_lb != c:
                    y[node] = best_lb
                    changed += 1

        if changed == 0:
            break
    return y

def warmup_factor_epoch(epoch, start_epoch, end_epoch):
    if epoch <= start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    t = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
    return t

def set_tv_weight_on_model(model, weight):
    setattr(model, "lambda_tv", float(weight))

def set_balance(model, args, active: bool):
    gb = args.gamma_balance
    model.gamma_balance = (gb if active else 0.0)

@torch.no_grad()
def compute_se_spatial(S: torch.Tensor,
                       edge_index: torch.Tensor,
                       edge_weight: torch.Tensor) -> float:
    """
    Structural entropy (2-layer) computed on the *spatial* graph only.
    Mirrors model.calculate_se_loss() but uses (edge_index, edge_weight)
    passed in (no attribute edges).
    """
    device = S.device
    dtype  = S.dtype
    N, K   = S.shape
    i, j   = edge_index[0], edge_index[1]
    w      = edge_weight

    # degrees and volumes
    deg   = scatter(w, i, dim=0, dim_size=N, reduce='sum')                         # vol at layer 2
    vol_G = deg.sum()
    eps   = torch.tensor(1e-12, device=device, dtype=dtype)

    vol1 = S.t() @ deg                                                   # (K,)
    vol0 = vol_G                                                         # scalar

    # -------- layer k = 1 (clusters) --------
    # parent volume of each cluster is the graph volume (2-layer tree)
    vol_parent1 = torch.full_like(vol1, vol0)
    # internal weight per cluster: sum_{(u,v)} w_uv * <S_u, S_v>
    weight_sum1 = (w.view(-1, 1) * (S[i] * S[j])).sum(dim=0)             # (K,)
    delta_vol1  = vol1 - weight_sum1
    log_ratio1  = torch.log2((vol1 + eps) / (vol_parent1 + eps))
    term1       = torch.dot(delta_vol1, log_ratio1)

    # -------- layer k = 2 (nodes) -----------
    vol2 = deg                                                            # (N,)
    # parent volume of each node: (S @ vol1)[u]
    vol_parent2 = S @ vol1                                                # (N,)
    # self-loop weight per node (spatial graph includes I)
    mask_diag   = (i == j)
    w_self      = scatter(w[mask_diag], i[mask_diag], dim=0, dim_size=N, reduce='sum')
    delta_vol2  = vol2 - w_self
    log_ratio2  = torch.log2((vol2 + eps) / (vol_parent2 + eps))
    term2       = torch.dot(delta_vol2, log_ratio2)

    se = -(term1 + term2) / (vol_G + eps)
    return float(se.detach().cpu())

@torch.no_grad()
def compute_edge_agreement_scores(S: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  edge_weight: torch.Tensor,
                                  hard_pred: torch.Tensor = None) -> tuple[float, float]:
    """
    EAS_soft: weighted mean of <S_i, S_j> over spatial edges (no self-loops).
    EAS_hard: weighted fraction of edges whose endpoints share the same hard label.
    """
    device = S.device
    i, j   = edge_index[0], edge_index[1]
    w      = edge_weight
    mask   = (i != j)                       # drop self-loops (builder adds them)
    i, j   = i[mask], j[mask]
    w      = w[mask]

    # soft agreement
    soft_sim = (S[i] * S[j]).sum(dim=1)     # (E,)
    eas_soft = (w * soft_sim).sum() / (w.sum() + 1e-12)

    # hard agreement
    if hard_pred is None:
        hard_pred = S.argmax(dim=1)
    eq        = (hard_pred[i] == hard_pred[j])
    eas_hard  = (w[eq].sum()) / (w.sum() + 1e-12)

    return float(eas_soft.detach().cpu()), float(eas_hard.detach().cpu())

def _minmax01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    return (x - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(x)

def orient_pseudotime_by_root_indices(t_node, root_indices, scale=True):
    """
    Orient t so that the provided root nodes have *low* pseudotime.
    root_indices: 1D array/list of node indices (int).
    """
    t = np.asarray(t_node, dtype=np.float64).copy()
    roots = np.asarray(root_indices, dtype=int)
    sign = -1.0 if t[roots].mean() > t.mean() else 1.0
    t = sign * t
    return _minmax01(t) if scale else t

def orient_pseudotime_by_pred_cluster(
    t_node,
    pred_labels,
    root_cluster,
    scale: bool = True,
):
    """
    Orient node-level pseudotime so that the given *predicted cluster*
    becomes the root (lowest pseudotime).

    Parameters
    ----------
    t_node : array-like (N,) or torch.Tensor
        Node-level pseudotime (e.g., from model.compute_spectral_pseudotime).
    pred_labels : array-like (N,) or torch.Tensor
        Predicted cluster label per node (typically ints from S.argmax()).
    root_cluster : int or str
        The SINGLE predicted cluster id to be treated as the root.
    scale : bool, default=True
        If True, min–max scale to [0, 1] after orientation.

    Returns
    -------
    np.ndarray (N,)
        Oriented (and optionally scaled) pseudotime with the chosen cluster low.
    """

    # to numpy 1D
    def _to_np1(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().ravel()
        return np.asarray(x).ravel()

    t = _to_np1(t_node).astype(np.float64, copy=True)
    labs = _to_np1(pred_labels)

    # build mask for the single cluster id (supports int or str labels)
    mask = (labs == root_cluster)
    if not np.any(mask):
        raise ValueError(f"No nodes found for root cluster '{root_cluster}'.")

    root_indices = np.where(mask)[0]
    # reuse existing orientation-by-indices helper for consistent behavior
    return orient_pseudotime_by_root_indices(t, root_indices, scale=scale)

def orient_pseudotime_by_spatial_anchor(t_node, coords, anchor="north", k=200, scale=True):
    """
    Orient by a spatial anchor:
      - "north": smallest y; "south": largest y; "west": smallest x; "east": largest x
    Uses the k most-extreme spots to make the anchor robust.
    coords: (N,2) np.ndarray or array-like in pixel/um space (adata.obsm['spatial']).
    """
    t = np.asarray(t_node, dtype=np.float64).copy()
    XY = np.asarray(coords)
    assert XY.shape[1] >= 2, "coords must be (N,2)"

    N = XY.shape[0]
    k = max(1, min(k, N))
    if anchor.lower() == "north":   # small y
        idx = np.argsort(XY[:, 1])[:k]
    elif anchor.lower() == "south": # large y
        idx = np.argsort(-XY[:, 1])[:k]
    elif anchor.lower() == "west":  # small x
        idx = np.argsort(XY[:, 0])[:k]
    elif anchor.lower() == "east":  # large x
        idx = np.argsort(-XY[:, 0])[:k]
    else:
        raise ValueError("anchor must be one of {'north','south','west','east'}")

    sign = -1.0 if t[idx].mean() > t.mean() else 1.0
    t = sign * t
    return _minmax01(t) if scale else t


# ------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------
def plot_pseudotime_spatial(
    adata,
    obs_key="pseudotime",
    cmap="viridis",
    s=None,
    alpha=1.0,
    invert_y=True,
    title=None,
    save_path=None,
    dpi=100,
    ax=None,
    cbar_labelsize=12,
    cbar_ticksize=10,
    cbar_title=None,
):
    """
    Scatter-plot pseudotime on tissue coordinates.

    Matched styling with the revised cluster plotting:
    - adaptive point size when s=None
    - equal aspect ratio
    - optional y-axis inversion for Visium-like coordinates
    - axis turned off
    - safe save_path directory creation
    """

    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")
    if obs_key not in adata.obs:
        raise ValueError(f"adata.obs['{obs_key}'] not found.")

    coords = np.asarray(adata.obsm["spatial"])
    t = np.asarray(adata.obs[obs_key].values, dtype=float)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must be an array of shape (N, 2) or (N, >=2).")

    # Drop rows with invalid coordinates or pseudotime values
    valid = (
        np.isfinite(coords[:, 0]) &
        np.isfinite(coords[:, 1]) &
        np.isfinite(t)
    )
    coords_plot = coords[valid, :2]
    t_plot = t[valid]

    n_obs = coords_plot.shape[0]

    # Matched adaptive sizing with the revised cluster plotting
    if s is None:
        s = float(np.clip(40000.0 / max(1, n_obs), 0.5, 10.0))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    sc_handle = ax.scatter(
        coords_plot[:, 0],
        coords_plot[:, 1],
        c=t_plot,
        s=s,
        alpha=alpha,
        cmap=cmap,
        edgecolors="none",
        rasterized=(n_obs > 50000),
    )

    if invert_y:
        ax.invert_yaxis()

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("" if title is None else title)

    cbar = fig.colorbar(sc_handle, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(obs_key if cbar_title is None else cbar_title, fontsize=cbar_labelsize)
    cbar.ax.tick_params(labelsize=cbar_ticksize)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        
    return ax


def _choose_spatial_axis(XY, axis="auto"):
    if axis in ("x", "y"):
        return axis
    span_x = np.nanpercentile(XY[:, 0], 95) - np.nanpercentile(XY[:, 0], 5)
    span_y = np.nanpercentile(XY[:, 1], 95) - np.nanpercentile(XY[:, 1], 5)
    return "y" if span_y >= span_x else "x"


def _auto_cluster_point_size(n_obs: int) -> float:
    #return float(np.clip(60000.0 / max(1, int(n_obs)), 1.0, 20.0))
    if n_obs < 10000:
        s = 50
    elif n_obs < 50000:
        s = 5
    elif n_obs < 150000:
        s = 1
    else:
        s = 0.5
    return s

def _scanpy_default_palette(n_categories: int):
    """
    Use Scanpy-style default categorical palettes.
    Falls back gracefully if a palette is unavailable.
    """
    if n_categories <= 20 and hasattr(sc.pl.palettes, "default_20"):
        return list(sc.pl.palettes.default_20[:n_categories])
    if n_categories <= 28 and hasattr(sc.pl.palettes, "default_28"):
        return list(sc.pl.palettes.default_28[:n_categories])
    if n_categories <= 102 and hasattr(sc.pl.palettes, "default_102"):
        return list(sc.pl.palettes.default_102[:n_categories])

    if hasattr(sc.pl.palettes, "default_102"):
        base = list(sc.pl.palettes.default_102)
    else:
        base = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    reps = int(np.ceil(n_categories / len(base)))
    return (base * reps)[:n_categories]


def _set_default_palette(adata, key):
    """
    Ensure a categorical palette exists for this key and return (categories, colors).
    If adata.uns already contains a matching '{key}_colors' entry, keep it.
    Otherwise, assign a default Scanpy-like palette.
    """
    adata.obs[key] = adata.obs[key].astype("category")
    cats = list(adata.obs[key].cat.categories)
    color_key = f"{key}_colors"
    colors = list(adata.uns.get(color_key, []))
    if len(colors) != len(cats):
        colors = _scanpy_default_palette(len(cats))
        adata.uns[color_key] = colors
    return cats, colors


def _category_centroids(adata, key):
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")

    adata.obs[key] = adata.obs[key].astype("category")
    cats = list(adata.obs[key].cat.categories)
    XY = np.asarray(adata.obsm["spatial"])
    vals = np.asarray(adata.obs[key].astype(object).values)

    centroids = []
    for c in cats:
        mask = (vals == c)
        centroids.append(np.nanmean(XY[mask], axis=0))
    return cats, np.vstack(centroids)


def relabel_pred_to_spatial_order(adata, pred_key="pred_region", axis="y"):
    """
    Fallback for unlabeled mode:
    - order predicted clusters by spatial centroid
    - rename them to 0..K-1
    - preserve existing colors if available; otherwise assign a default palette
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")

    XY = np.asarray(adata.obsm["spatial"])
    axis = _choose_spatial_axis(XY, axis)

    adata.obs[pred_key] = adata.obs[pred_key].astype("category")
    cats_old = list(adata.obs[pred_key].cat.categories)

    colors_old = list(adata.uns.get(f"{pred_key}_colors", []))
    if len(colors_old) != len(cats_old):
        colors_old = _scanpy_default_palette(len(cats_old))
    color_by_old = dict(zip(cats_old, colors_old))

    _, C = _category_centroids(adata, pred_key)
    a = np.array([0.0, 1.0]) if axis == "y" else np.array([1.0, 0.0])
    scores = C @ a
    order_spatial = np.argsort(scores)
    ordered_old = [cats_old[i] for i in order_spatial]

    mapping = {old: i for i, old in enumerate(ordered_old)}
    adata.obs[pred_key] = adata.obs[pred_key].cat.rename_categories(mapping)
    adata.obs[pred_key] = adata.obs[pred_key].cat.reorder_categories(
        list(range(len(ordered_old))), ordered=True
    )

    adata.uns[f"{pred_key}_colors"] = [color_by_old[old] for old in ordered_old]

    return {
        "pred_order_old": ordered_old,
        "pred_order_new": list(range(len(ordered_old))),
    }


def align_pred_colors_spatial(adata, label="Region", pred_key="pred_region", axis="auto"):
    """
    Labeled mode:
    - force GT annotation colors to the default Scanpy-like palette
    - match predicted clusters to GT colors by maximum overlap
    - order predicted clusters by matched GT order
    - rename predicted clusters to 0..K-1 in that matched order
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")
    if label not in adata.obs:
        raise ValueError(f"adata.obs['{label}'] not found.")

    XY = np.asarray(adata.obsm["spatial"])
    axis = _choose_spatial_axis(XY, axis)

    # Ground truth: force default palette
    adata.obs[label] = adata.obs[label].astype("category")
    gt_cats, gt_colors = _set_default_palette(adata, label)
    Kg = len(gt_cats)

    # Predictions
    adata.obs[pred_key] = adata.obs[pred_key].astype("category")
    pr_cats = list(adata.obs[pred_key].cat.categories)
    Kp = len(pr_cats)

    if Kg == 0 or Kp == 0:
        return {"pairing": [], "pred_order_old": [], "pred_order_new": []}

    # Overlap matrix: rows = pred, cols = GT
    pr_codes = adata.obs[pred_key].cat.codes.to_numpy()
    gt_codes = adata.obs[label].cat.codes.to_numpy()
    valid = (pr_codes >= 0) & (gt_codes >= 0)

    overlap = np.zeros((Kp, Kg), dtype=np.float64)
    np.add.at(overlap, (pr_codes[valid], gt_codes[valid]), 1.0)

    row_ind, col_ind = linear_sum_assignment(-overlap)

    # Map old predicted categories -> GT color
    color_by_old_pred = {}
    matched_pairs = []
    used_pred = set()
    used_gt = set()

    for pi, gi in zip(row_ind, col_ind):
        if overlap[pi, gi] > 0:
            color_by_old_pred[pr_cats[pi]] = gt_colors[gi]
            matched_pairs.append((pr_cats[pi], gt_cats[gi], int(overlap[pi, gi]), gi))
            used_pred.add(pr_cats[pi])
            used_gt.add(gi)

    # Order matched predicted categories by GT category order
    matched_pairs = sorted(matched_pairs, key=lambda x: x[3])
    ordered_old = [p for p, g, n, gi in matched_pairs]

    # Unmatched predicted categories: append by spatial order
    remaining = [c for c in pr_cats if c not in used_pred]
    if remaining:
        _, Cpred = _category_centroids(adata, pred_key)
        old_to_idx = {c: i for i, c in enumerate(pr_cats)}
        a = np.array([0.0, 1.0]) if axis == "y" else np.array([1.0, 0.0])
        scores = Cpred @ a
        remaining = sorted(remaining, key=lambda c: scores[old_to_idx[c]])

        unused_gt_colors = [gt_colors[i] for i in range(Kg) if i not in used_gt]
        fill_palette = unused_gt_colors if len(unused_gt_colors) > 0 else gt_colors

        for j, c in enumerate(remaining):
            color_by_old_pred[c] = fill_palette[j % len(fill_palette)]

        ordered_old.extend(remaining)

    # Rename predicted domains to 0..K-1 in the matched order
    mapping = {old: i for i, old in enumerate(ordered_old)}
    adata.obs[pred_key] = adata.obs[pred_key].cat.rename_categories(mapping)
    adata.obs[pred_key] = adata.obs[pred_key].cat.reorder_categories(
        list(range(len(ordered_old))), ordered=True
    )

    # Palette in the new category order 0..K-1
    adata.uns[f"{pred_key}_colors"] = [color_by_old_pred[old] for old in ordered_old]

    return {
        "pairing": [(p, g, n) for p, g, n, _ in matched_pairs],
        "pred_order_old": ordered_old,
        "pred_order_new": list(range(len(ordered_old))),
    }


def plot_cluster_spatial(
    adata,
    save_path=None,
    s=None,
    legend_fontsize=10,
    label="Region",
    pred_key="pred_region",
    eval_mode=1,
    legend_title_gt="Annotation",
    legend_title_pred="Domains",
    legend_titlesize=None,
    legend_loc="right margin",
    alpha=1.0,
    dpi=100,
    invert_y=True,
):
    """
    Plot annotation and predicted domains with:
    - default Scanpy-like palette (only assigned when a palette is absent)
    - matched GT/pred colors when labels are available
    - predicted domains relabeled to 0..K-1
    - optional y-axis inversion controlled by `invert_y`
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")

    XY = np.asarray(adata.obsm["spatial"])

    if eval_mode == 1 and label in adata.obs and not adata.obs[label].isna().all():
        align_pred_colors_spatial(adata, label=label, pred_key=pred_key, axis="auto")
        adata.obs[label] = adata.obs[label].astype("category")
    else:
        relabel_pred_to_spatial_order(adata, pred_key=pred_key, axis="auto")

    adata.obs[pred_key] = adata.obs[pred_key].astype("category")
    adata.obs["x_coord"] = XY[:, 0]
    adata.obs["y_coord"] = XY[:, 1]

    if s is None:
        s = _auto_cluster_point_size(adata.n_obs)

    def _format_ax(ax):
        ax.set_aspect("equal")
        if invert_y:
            ax.invert_yaxis()
        ax.set_axis_off()
        ax.set_title(None)

    def _set_legend_title(ax, title):
        leg = ax.get_legend()
        if leg is None:
            leg = getattr(ax, "legend_", None)
        if leg is None:
            handles, legend_labels = ax.get_legend_handles_labels()
            if handles:
                leg = ax.legend(
                    handles,
                    legend_labels,
                    title=title,
                    fontsize=legend_fontsize,
                    frameon=False,
                    loc="upper right",
                )
        if leg is not None:
            leg.set_title(title)
            ts = legend_titlesize if legend_titlesize is not None else legend_fontsize
            leg.get_title().set_fontsize(ts)

    if eval_mode == 1 and label in adata.obs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        sc.pl.scatter(
            adata, x="x_coord", y="y_coord", color=label,
            size=s, alpha=alpha, frameon=False,
            ax=axes[0], show=False,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc,
        )
        sc.pl.scatter(
            adata, x="x_coord", y="y_coord", color=pred_key,
            size=s, alpha=alpha, frameon=False,
            ax=axes[1], show=False,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc,
        )

        _format_ax(axes[0])
        _format_ax(axes[1])
        _set_legend_title(axes[0], legend_title_gt)
        _set_legend_title(axes[1], legend_title_pred)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        sc.pl.scatter(
            adata, x="x_coord", y="y_coord", color=pred_key,
            size=s, alpha=alpha, frameon=False,
            ax=ax, show=False,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc,
        )

        _format_ax(ax)
        _set_legend_title(ax, legend_title_pred)

    plt.tight_layout(pad=3.0)
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)