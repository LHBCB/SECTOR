import torch
import torch.nn.functional as F
import numpy as np
import random
from scipy.sparse.csgraph import connected_components
import os
import matplotlib.pyplot as plt

from torch_geometric.utils import scatter

import re
import scanpy as sc
from .build_graph import build_spatial_graph
from torch_geometric.data import Data
import scipy.sparse
from scipy.optimize import linear_sum_assignment

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def adata_preprocess(adata, min_cells=5, min_counts=10, target_sum=1e4, n_comps=20, n_top_genes=2000, random_seed=42):
    #print(f"Num of HVGs: {n_top_genes}\nNum of PCs: {n_comps}")
    adata.layers['counts'] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='counts', n_top_genes=n_top_genes)
    #adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver='auto', zero_center=True, random_state=random_seed)
    return adata

def read_st_data(args):
    adata = sc.read_h5ad(f'{args.dataset_path}/{args.dataset}/{args.slice}.h5ad')

    eval_mode = int(getattr(args, "eval_mode", 1)) == 1

    # In supervised eval_mode, keep only labeled spots
    if eval_mode:
        if args.label not in adata.obs:
            raise ValueError(f"Label column '{args.label}' not found in adata.obs.")
        adata = adata[~adata.obs[args.label].isna()].copy()
    # In unsupervised eval_mode, keep all spots (no filtering)

    graph_dict = build_spatial_graph(adata, n=args.k_s, weight_mode=args.weight_mode)
    edge_index = graph_dict['adj_label'].indices()
    edge_weight = graph_dict['adj_label'].values().float()

    adata = adata_preprocess(
        adata,
        n_comps=args.n_comps,
        n_top_genes=args.n_top_genes,
        random_seed=args.seed
    )

    x = torch.from_numpy(adata.obsm['X_pca']).float()

    if eval_mode:
        # Build integer labels only when evaluating with ground truth
        labels = adata.obs[args.label].values
        # Handle pandas Categorical or plain arrays
        categories = getattr(labels, "categories", np.unique(labels))
        label_to_index = {label: idx for idx, label in enumerate(categories)}
        index_list = [label_to_index[lbl] for lbl in labels]
        y = torch.tensor(index_list, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    else:
        # Strictly unsupervised: do NOT attach y
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
    import numpy as np
    import torch

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

def align_pred_colors_spatial(
    adata,
    label="Region",
    pred_key="pred_region",
    axis="auto"   # "y" for top→bottom, "x" for left→right, or "pc1" to auto-pick main axis from GT
):
    # --- sanity ---
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")
    XY = adata.obsm["spatial"]
    gt = adata.obs[label].astype("category")
    pr = adata.obs[pred_key].astype("category")
    adata.obs[label] = gt
    adata.obs[pred_key] = pr
    gt_cats  = list(gt.cat.categories)
    pr_cats  = list(pr.cat.categories)
    Kg, Kp   = len(gt_cats), len(pr_cats)

    # --- get GT colors as a dict(cat -> color) ---
    gt_colors = adata.uns.get(f"{label}_colors")
    if gt_colors is None or len(gt_colors) != Kg:
        # fall back to a default palette if GT didn’t have one yet
        import scanpy as sc
        pal = sc.pl.palettes.default_20
        gt_colors = [pal[i % len(pal)] for i in range(Kg)]
        adata.uns[f"{label}_colors"] = gt_colors
    color_by_gt = dict(zip(gt_cats, gt_colors))

    # --- centroids for each category ---
    def centroids(cats, key):
        C = []
        vals = adata.obs[key].values
        for c in cats:
            m = vals == c
            C.append(np.nanmean(XY[m], axis=0))
        return np.vstack(C)

    Cg = centroids(gt_cats, label)   # (Kg, 2)
    Cp = centroids(pr_cats, pred_key) # (Kp, 2)

    # --- choose a 1D ordering axis ---
    if axis == "y":      # vertical ordering (top→bottom)
        a = np.array([0.0, 1.0])
    elif axis == "x":    # horizontal ordering (left→right)
        a = np.array([1.0, 0.0])
    else:                # "auto": main axis from GT centroids
        X = Cg - np.nanmean(Cg, axis=0, keepdims=True)
        # simple SVD for the first principal direction
        _, _, Vt = np.linalg.svd(np.nan_to_num(X), full_matrices=False)
        a = Vt[0]

    # --- project centroids and get ranks along that axis ---
    sg = (Cg @ a)  # GT scores
    sp = (Cp @ a)  # Pred scores

    rg = np.argsort(np.argsort(sg))  # ranks 0..Kg-1
    rp = np.argsort(np.argsort(sp))  # ranks 0..Kp-1

    # --- assign predicted clusters to GT colors by matching ranks (Hungarian on |rank diff|) ---
    cost = np.abs(rp[:, None] - rg[None, :])  # (Kp, Kg)
    row_ind, col_ind = linear_sum_assignment(cost)

    # build color list for predicted categories (in *pred category order*)
    pred_colors = ["#808080"] * Kp
    for pi, gi in zip(row_ind, col_ind):
        pred_colors[pi] = color_by_gt[gt_cats[gi]]

    # if there are more pred clusters than GT, fill remaining by sampling GT palette in order
    if any(c == "#808080" for c in pred_colors):
        order_gt_by_axis = list(np.argsort(sg))
        palette_sorted   = [gt_colors[i] for i in order_gt_by_axis]
        idx = np.clip(np.round(np.linspace(0, len(palette_sorted)-1, Kp)).astype(int), 0, len(palette_sorted)-1)
        for i, c in enumerate(pred_colors):
            if c == "#808080":
                pred_colors[i] = palette_sorted[idx[i]]

    # reorder predicted categories themselves to appear in spatial order in legends/plots
    pred_order = [pr_cats[i] for i in np.argsort(sp)]
    adata.obs[pred_key] = adata.obs[pred_key].cat.reorder_categories(pred_order)

    # write colors into .uns in the same order as adata.obs[pred_key].cat.categories
    adata.uns[f"{pred_key}_colors"] = [pred_colors[pr_cats.index(c)] for c in pred_order]

    return {
        "pred_order": pred_order,
        "pairing": list(zip([pr_cats[i] for i in row_ind], [gt_cats[j] for j in col_ind]))
    }


def relabel_pred_to_spatial_order(adata, pred_key="pred_region", axis="y"):
    """
    Rename predicted categories to 0..K-1 according to spatial order along `axis`
    ('y' = top→bottom, 'x' = left→right), while preserving colors.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")
    XY = adata.obsm["spatial"]

    pr = adata.obs[pred_key].astype("category")
    cats_old = list(pr.cat.categories)
    colors_old = adata.uns.get(f"{pred_key}_colors")
    if colors_old is None or len(colors_old) != len(cats_old):
        raise ValueError(f"{pred_key}_colors missing or length mismatch.")
    color_by_old = dict(zip(cats_old, colors_old))

    # centroids per predicted category
    cent = []
    for c in cats_old:
        m = (adata.obs[pred_key].values == c)
        cent.append(np.nanmean(XY[m], axis=0))
    C = np.vstack(cent)

    # pick axis
    a = np.array([0.0, 1.0]) if axis == "y" else np.array([1.0, 0.0])
    scores = C @ a
    order_spatial = np.argsort(scores)         # indices of cats_old in spatial order

    # rename: old_cat -> new integer 0..K-1 following spatial order
    mapping = {cats_old[i]: i for i in order_spatial}
    adata.obs[pred_key] = pr.cat.rename_categories(mapping)

    # ensure legend order is 0..K-1
    K = len(cats_old)
    new_order = list(range(K))
    adata.obs[pred_key] = adata.obs[pred_key].cat.reorder_categories(new_order, ordered=True)

    # rebuild palette in 0..K-1 order using the old colors of the spatially ordered cats
    adata.uns[f"{pred_key}_colors"] = [color_by_old[cats_old[i]] for i in order_spatial]

def plot_pseudotime_spatial(adata, obs_key="pseudotime", cmap="viridis",
                            s=3, alpha=1.0, invert_y=True, title=None,
                            save_path=None, dpi=200, ax=None, cbar_labelsize=6, cbar_ticksize=6):
    """
    Scatter-plot pseudotime on tissue coordinates.
    - obs_key: column in adata.obs with pseudotime in [0,1]
    - s: point size; if None, chosen adaptively based on N
    - invert_y: Visium-like coordinates have origin at top-left; invert to match tissue
    """
    import matplotlib.pyplot as plt

    if "spatial" not in adata.obsm_keys():
        raise ValueError("adata.obsm['spatial'] not found.")
    coords = np.asarray(adata.obsm["spatial"])
    t = np.asarray(adata.obs[obs_key].values, dtype=float)
    N = coords.shape[0]
    s = s if s is not None else max(1.0, 20000.0 / max(1, N))

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=dpi)

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=t, s=s, alpha=alpha, cmap=cmap, edgecolors='none')
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect('equal')
    #ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    ax.set_axis_off() 
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    
    cbar.set_label(obs_key, fontsize=cbar_labelsize)
    cbar.ax.tick_params(labelsize=cbar_ticksize)
    
    if save_path:
        os.makedirs("figures", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    return ax

def plot_cluster_spatial(
    adata,
    save_path=None,
    s=50,
    legend_fontsize=10,
    label='Region',
    pred_key='pred_region',
    eval_mode=1,
    legend_title_gt="Annotation",
    legend_title_pred="Domains",
    legend_titlesize=None,
    legend_loc="right margin",
):

    align_pred_colors_spatial(adata, label=label, pred_key=pred_key, axis="auto")
    relabel_pred_to_spatial_order(adata, pred_key=pred_key, axis="y")

    # Ensure categorical dtypes
    adata.obs[label] = adata.obs[label].astype("category")
    adata.obs["pred_region"] = adata.obs[pred_key].astype("category")

    # Pull out spatial coordinates
    adata.obs['x_coord'] = adata.obsm['spatial'][:, 0]
    adata.obs['y_coord'] = adata.obsm['spatial'][:, 1]

    def _format_ax(ax):
        ax.set_aspect("equal")
        ax.invert_yaxis()        # matches Visium orientation
        ax.set_axis_off() 
        ax.set_title(None)  

    def _set_legend_title(ax, title):
        # Try to get the legend Scanpy created; if missing, build one from handles.
        leg = ax.get_legend()
        if leg is None:
            leg = getattr(ax, "legend_", None)
        if leg is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                leg = ax.legend(handles, labels, title=title,
                                fontsize=legend_fontsize, frameon=False, loc="upper right")
        if leg is not None:
            leg.set_title(title)
            # Title font size; default to legend_fontsize if not provided
            ts = legend_titlesize if legend_titlesize is not None else legend_fontsize
            leg.get_title().set_fontsize(ts)

    if eval_mode == 1:
        # Two-panel figure: ground-truth + prediction
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        sc.pl.scatter(
            adata, x="x_coord", y="y_coord", color=label,
            size=s, frameon=False,
            ax=axes[0], show=False,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc
        )
        sc.pl.scatter(
            adata, x="x_coord", y="y_coord", color="pred_region",
            size=s, frameon=False,
            ax=axes[1], show=False,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc
        )

        _format_ax(axes[0])
        _format_ax(axes[1])

        # Set legend titles
        _set_legend_title(axes[0], legend_title_gt)
        _set_legend_title(axes[1], legend_title_pred)

    else:
        # Single-panel figure: prediction only
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        sc.pl.scatter(
            adata, x="x_coord", y="y_coord", color="pred_region",
            size=s, frameon=False,
            ax=ax, show=False,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc
        )

        _format_ax(ax)
        _set_legend_title(ax, legend_title_pred)

    plt.tight_layout(pad=3.0)
    if save_path:
        os.makedirs("figures", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
