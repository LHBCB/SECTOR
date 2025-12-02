import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors

from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import scatter

from .utility.utils import select_activation, g_from_torchsparse

# ------------------------- Graph utilities -------------------------

@torch.no_grad()
def KNN_weighted(x: torch.Tensor,
                 k: int,
                 metric: str = "cosine",
                 sigma: float = None,
                 chunk_size: int = 2048) -> torch.Tensor:
    """
    GPU KNN using torch.cdist + topk

    - metric="cosine": neighbors by cosine distance; weights = cosine similarity
                       (clipped to [0,1]); implemented via cdist on L2-normalized vectors.
    - else (Euclidean): neighbors by L2 distance; weights = exp(-d^2/(2*sigma^2)) if sigma
                        is given, otherwise 1.0
    - Returns a symmetrized sparse adjacency: 0.5 * (A + A^T).
    """
    assert x.dim() == 2, "x must be [N, D]"
    N, _ = x.shape
    device, dtype = x.device, x.dtype
    if N == 0 or k <= 0:
        return torch.sparse_coo_tensor(
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, dtype=dtype, device=device),
            (N, N),
            device=device
        ).coalesce()

    k = min(k, max(1, N - 1))

    # Database tensor
    xb = F.normalize(x, p=2, dim=1, eps=1e-12) if metric == "cosine" else x

    rows_all, cols_all, vals_all = [], [], []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        q = xb[start:end]                 # [B, D]
        B = q.shape[0]

        # Distances
        dist = torch.cdist(q, xb, p=2)    # [B, N]
        # Exclude self within this block
        row_idx = torch.arange(start, end, device=device)
        dist[torch.arange(B, device=device), row_idx] = float("inf")

        # Top-k nearest (smallest distance)
        dvals, idx = torch.topk(dist, k, dim=1, largest=False, sorted=False)

        # Weights
        if metric == "cosine":
            # For unit vectors: cos_sim = 1 - (||u-v||^2)/2 = 1 - (d^2)/2
            w = (1.0 - 0.5 * (dvals ** 2)).clamp_(min=0.0, max=1.0)
        else:
            if sigma is not None:
                w = torch.exp(-(dvals ** 2) / (2.0 * (sigma ** 2)))
            else:
                w = torch.ones_like(dvals, dtype=dtype, device=device)

        rows = torch.arange(start, end, device=device).repeat_interleave(k)
        cols = idx.reshape(-1)
        vals = w.reshape(-1).to(dtype)

        rows_all.append(rows)
        cols_all.append(cols)
        vals_all.append(vals)

    rows = torch.cat(rows_all, dim=0)
    cols = torch.cat(cols_all, dim=0)
    vals = torch.cat(vals_all, dim=0)

    A = torch.sparse_coo_tensor(
        torch.stack([rows, cols], dim=0),
        vals,
        (N, N),
        device=device
    ).coalesce()

    # Symmetrize by averaging
    A = (A + A.transpose(0, 1)).coalesce()
    if A._nnz() > 0:
        A = torch.sparse_coo_tensor(
            A.indices(), A.values() * 0.5, A.size(), device=device
        ).coalesce()

    return A

def rescale_total_mass(A_sparse: torch.Tensor, target_sum: torch.Tensor):
    A = A_sparse.coalesce()
    cur = A.values().sum()
    scale = (target_sum / (cur + 1e-12)).clamp(min=0.1, max=10.0)
    return torch.sparse_coo_tensor(A.indices(), A.values() * scale, A.size()).coalesce()

# --------------------- TV on soft assignments ----------------------

def spatial_tv_loss(S: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
    i, j = edge_index
    diff = S[i] - S[j]           # (E, C)
    l1   = diff.abs().sum(dim=1) # (E,)
    num  = (edge_weight * l1).sum()
    den  = edge_weight.sum().clamp_min(1e-9)
    return num / den

# ---------------- Spectral pseudotime (unoriented) -----------------

def _compute_cluster_graph_dense(S: torch.Tensor, A_sparse: torch.Tensor) -> torch.Tensor:
    AS = torch.sparse.mm(A_sparse, S)   # (N,K)
    A_c = S.T @ AS                      # (K,K)
    A_c = 0.5 * (A_c + A_c.T)
    A_c = torch.clamp(A_c, min=0.0)
    return A_c

def _laplacian(A_c: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    d = A_c.sum(dim=1)
    if normalized:
        eps = 1e-12
        inv_sqrt = torch.pow(d + eps, -0.5)
        Dm1_2_A = inv_sqrt[:, None] * A_c * inv_sqrt[None, :]
        L = torch.eye(A_c.shape[0], device=A_c.device, dtype=A_c.dtype) - Dm1_2_A
    else:
        L = torch.diag(d) - A_c
    return L

def _drop_diag(A: torch.Tensor) -> torch.Tensor:
    return A - torch.diag_embed(A.diag())

@torch.no_grad()
def _components_from_dense(A: torch.Tensor) -> list[torch.Tensor]:
    """
    Connected components of a small dense symmetric adjacency (K x K).
    Returns list of index tensors (on the same device).
    """
    K = A.shape[0]
    if K == 0:
        return []
    adj = (A > 0).to(torch.bool)
    # ensure symmetry
    adj = torch.logical_or(adj, adj.T)

    seen = torch.zeros(K, dtype=torch.bool, device=A.device)
    comps = []
    for u in range(K):
        if seen[u]:
            continue
        q = [u]
        seen[u] = True
        comp = [u]
        while q:
            v = q.pop()
            nbrs = torch.nonzero(adj[v]).view(-1)
            for w in nbrs.tolist():
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
                    comp.append(w)
        comps.append(torch.tensor(comp, device=A.device, dtype=torch.long))
    return comps

@torch.no_grad()
def compute_tau_fiedler_from_SA(
    S: torch.Tensor,
    A_sparse: torch.Tensor,
    normalized: bool = True,
    zscore: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (tau, A_c) where tau is the per-component Fiedler vector (2nd eigen)
    of the Laplacian of A_c = S^T A S, after dropping self-loops on A_c.
    """
    A_c = _compute_cluster_graph_dense(S, A_sparse)

    # --- NEW: drop diagonal (self-loops) on the cluster graph ---
    A_c = _drop_diag(A_c).clamp_min(0.0)

    K = S.shape[1]
    tau = torch.zeros(K, device=S.device, dtype=S.dtype)

    comps = _components_from_dense(A_c)
    if len(comps) == 0:
        return tau, A_c

    for comp in comps:
        if comp.numel() == 1:
            tau[comp] = 0.0
            continue
        Ac = A_c.index_select(0, comp).index_select(1, comp)
        Lc = _laplacian(Ac, normalized=normalized)
        evals, evecs = torch.linalg.eigh(Lc)  # ascending
        # 2nd eigenvector if present; otherwise zeros
        if evecs.shape[1] >= 2:
            fv = evecs[:, 1]
        else:
            fv = torch.zeros_like(comp, dtype=S.dtype, device=S.device)
        tau[comp] = fv

    if zscore:
        tau = (tau - tau.mean()) / (tau.std() + 1e-12)
    return tau, A_c

@torch.no_grad()
def node_pseudotime_from_S(
    S: torch.Tensor,
    tau: torch.Tensor,
    A_sparse: Optional[torch.Tensor] = None,
    post_smooth: float = 0.0,
    iters: int = 5,
    keep_cluster_means: bool = True,
) -> torch.Tensor:
    """
    Back-project τ to nodes: t = S @ τ, optionally graph-smooth on A_sparse.
    """
    t = S @ tau  # (N,)
    if A_sparse is None or post_smooth <= 0.0:
        return t.view(-1)

    A = A_sparse.coalesce()
    i, j = A.indices()
    w = A.values()
    eps = 1e-12
    t = t.view(-1)

    for _ in range(int(iters)):
        msg = torch.zeros_like(t).index_add(0, i, w * t[j])
        deg = torch.zeros_like(t).index_add(0, i, w)
        nbr_avg = msg / (deg + eps)
        t_new = (1.0 - post_smooth) * t + post_smooth * nbr_avg

        if keep_cluster_means:
            sumS = S.sum(0) + eps
            mean_curr = (S.T @ t_new) / sumS
            mean_back = (S.T @ (S @ tau)) / sumS
            delta = S @ (mean_back - mean_curr)
            t_new = t_new + delta

        t = t_new

    return t.view(-1)

# ---------------------------- Layers --------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, activation=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = select_activation(activation)
    def forward(self, x):
        x = self.fc1(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class WeightedMeanConv(MessagePassing):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, x, edge_index, edge_weight):
        assert edge_weight is not None, "edge_weight required for WeightedMeanConv"
        h = self.lin(x)
        src, dst = edge_index[0], edge_index[1]
        deg_w = scatter(edge_weight, dst, dim=0, dim_size=x.size(0), reduce='sum').clamp_min(1e-12)
        norm = edge_weight / deg_w[dst]
        return self.propagate(edge_index, x=h, w=norm)
    def message(self, x_j, w):
        return x_j * w.view(-1, 1)

class GCN_layer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, att=False):
        super().__init__()
        self.activation = select_activation(activation)
        self.att = att
        self.use_edge_attr = False
        if att:
            try:
                self.conv = GATConv(in_dim, out_dim, heads=1, concat=False,
                                    negative_slope=0.2, dropout=0.0, edge_dim=1)
                self.use_edge_attr = True
            except TypeError:
                self.conv = GATConv(in_dim, out_dim, heads=1, concat=False,
                                    negative_slope=0.2, dropout=0.0)
        else:
            self.conv = WeightedMeanConv(in_dim, out_dim)
    def forward(self, x, edge_index, edge_weight):
        if self.att and isinstance(self.conv, GATConv):
            edge_attr = edge_weight.unsqueeze(-1) if (self.use_edge_attr and edge_weight is not None) else None
            h = self.conv(x, edge_index, edge_attr=edge_attr)
        else:
            h = self.conv(x, edge_index, edge_weight=edge_weight)
        return self.activation(h) if self.activation else h

class GAT_layer(nn.Module):
    def __init__(self, embed_dim, num_cluster, k, dropout=0.0, activation=None):
        super().__init__()
        # GCN_emb for computing cluster-level embeddings (for reporting/visualisation purposes; not used for clustering/psuedotime)
        self.GCN_emb = GCN_layer(embed_dim, embed_dim, activation, att=False)
        self.GCN_ass = GCN_layer(embed_dim, num_cluster, activation, att=True)
        self.num_cluster = num_cluster
        self.k = k
    def forward(self, x, edge_index, edge_weight, adj_g_sparse):
        # x is the node embeddings at the current layer
        h = self.GCN_emb(x, edge_index, edge_weight)
        s = torch.softmax(self.GCN_ass(x, edge_index, edge_weight), dim=-1)
        e = s.t() @ h
        return h, e, s

# --------------------------- SEModel core --------------------------------

class SEModel(nn.Module):
    def __init__(self, args, feature, device):
        super().__init__()
        self.num_nodes = feature.shape[0]
        self.input_dim = feature.shape[-1]
        self.embed_dim = args.embed_dim
        self.activation = args.activation
        self.height = 2  # fixed 2-layer SE

        self.num_clusters = args.num_clusters

        self.mlp = MLP(self.input_dim, self.embed_dim, self.embed_dim)
        self.gnn = GCN_layer(self.input_dim, self.embed_dim, self.activation, att=False)

        self.assignlayers = nn.ModuleList([]) # using list for stacking more layers in the future
        self.assignlayers.append(GAT_layer(self.embed_dim, self.num_clusters, args.k, args.dropout, self.activation))
        
        self.device = device
        self.beta_f = args.beta_f
        self.k      = args.k

        # regularizers
        self.lambda_tv = getattr(args, "lambda_tv", 1.0)
        self.gamma_balance = getattr(args, "gamma_balance", 0.5)
        self.balance_mode  = getattr(args, "balance_mode", "volume")
        self.target_usage  = getattr(args, "target_usage", None)

        # caches for TV & pseudotime smoothing
        self._tv_ei = None
        self._tv_ew = None
        self._A_for_traj = None
        
        self.attr_metric = 'cosine' # fixed 'cosine', can also be 'euclidean'
        self.attr_sigma  = 5.0 # fixed 5.0
        
        self.pseudotime_spectral: Optional[torch.Tensor] = None

    def hard(self, s_dic):
        # Only keep the node->cluster hard assignment; avoid N×N tensors
        S = s_dic[self.height]  # (N, K)
        idx = S.argmax(dim=1)
        H = torch.zeros_like(S)
        H[torch.arange(S.shape[0], device=S.device), idx] = 1
        self.hard_dic = {1: H}
    
    def forward(self, adj_g_sparse, feature, degree=None):
        feature = feature.to(self.device)
        adj_g_sparse = adj_g_sparse.coalesce().to(self.device)

        # store structural edges for TV & spectral smoothing
        self._tv_ei = adj_g_sparse.indices()
        self._tv_ew = adj_g_sparse.values()
        self._A_for_traj = adj_g_sparse

        # attribute graph
        E_expr = self.mlp(feature)
        adj_f_raw = KNN_weighted(E_expr, self.k, metric=self.attr_metric, sigma=self.attr_sigma).to(self.device)

        fused = (adj_g_sparse + self.beta_f * adj_f_raw).coalesce()

        # embedding on fused graph
        g = g_from_torchsparse(fused)
        e = self.gnn(feature, g.edge_index, g.edge_weight)

        s_dic = {}
        tree_node_embed_dic = {self.height: e.to(self.device)}
        g_dic = {self.height: g}

        # assignment at the finest level
        adj_g_level = adj_g_sparse
        for i, layer in enumerate(self.assignlayers):
            # For the current implementation (height=2) there is a single assignment
            # layer that maps node embeddings directly to soft cluster assignments.
            h, e_coarse, s = layer(e, g.edge_index, g.edge_weight, adj_g_level)
            tree_node_embed_dic[self.height - i - 1] = e_coarse.to(self.device)
            s_dic[self.height - i] = s.to(self.device)

        s_dic[1] = torch.ones(s.shape[-1], 1, device=self.device)
        self.hard(s_dic)
        self.s_dic = s_dic
        self.g_dic = g_dic
        self.pseudotime_spectral = None
        return s_dic, tree_node_embed_dic, g_dic

    # ---------------------------- losses -----------------------------

    def calculate_se_loss(self):
        # fused graph at top level (height=2)
        g = self.g_dic[self.height]
        i, j = g.edge_index[0], g.edge_index[1]
        w = g.edge_weight
        N = self.num_nodes
        S = self.s_dic[self.height]              # (N, K)
        K = S.size(1)
        eps = 1e-12
    
        # degrees / volumes on fused graph
        deg   = scatter(w, i, dim=0, dim_size=N, reduce='sum')           # (N,)
        vol_G = deg.sum()
    
        # ---------- layer 1: clusters ----------
        vol1 = S.t() @ deg                                      # (K,)
        # parent volume is the graph volume at height-2 tree
        vol_parent1 = torch.full_like(vol1, vol_G)
    
        # internal weight per cluster: sum_e w_e * <S_i, S_j>, O(E*K) not E*N
        # (chunking is optional; keeps peak memory small on huge graphs)
        weight_sum1 = torch.zeros(K, device=S.device, dtype=S.dtype)
        chunk = 250_000
        for start in range(0, i.numel(), chunk):
            end = min(start + chunk, i.numel())
            si = S[i[start:end]]      # (B, K)
            sj = S[j[start:end]]      # (B, K)
            weight_sum1 += (w[start:end].view(-1, 1) * (si * sj)).sum(dim=0)
    
        delta_vol1 = vol1 - weight_sum1
        log_ratio1 = torch.log2((vol1 + eps) / (vol_parent1 + eps))
        term1 = torch.dot(delta_vol1, log_ratio1)
    
        # ---------- layer 2: nodes ----------
        vol2 = deg                                                        # (N,)
        vol_parent2 = S @ vol1                                            # (N,)
        # internal weight per node = self-loop weight (no E×N build!)
        mask_self = (i == j)
        w_self = scatter(w[mask_self], i[mask_self], dim=0, dim_size=N, reduce='sum')
        delta_vol2 = vol2 - w_self
        log_ratio2 = torch.log2((vol2 + eps) / (vol_parent2 + eps))
        term2 = torch.dot(delta_vol2, log_ratio2)
    
        se = -(term1 + term2) / (vol_G + eps)
        return se
    
    def calculate_tv_loss(self):
        if self.lambda_tv <= 0.0 or self._tv_ei is None:
            return torch.tensor(0.0, device=self.device)
        S = self.s_dic[self.height]
        return spatial_tv_loss(S, self._tv_ei, self._tv_ew)

    def calculate_balance_loss(self):
        if self.gamma_balance <= 0.0:
            return torch.tensor(0.0, device=self.device)
        S = self.s_dic[self.height]  # (N, K)
        K = S.size(1)
        if self.balance_mode == "volume":
            g = self.g_dic[self.height]
            deg = scatter(g.edge_weight, g.edge_index[0], dim_size=self.num_nodes, reduce='sum')
            
            p = (S.t() @ deg) / (deg.sum() + 1e-12)
        else:
            p = S.mean(dim=0)
        if self.target_usage is None:
            u = torch.full_like(p, 1.0 / K)
        else:
            u = torch.as_tensor(self.target_usage, dtype=p.dtype, device=p.device)
            u = u / (u.sum() + 1e-12)
        eps = 1e-9
        bal = torch.sum(p * (torch.log(p + eps) - torch.log(u + eps)))
        return bal

    def total_loss(self):
        se = self.calculate_se_loss()
        tv = self.calculate_tv_loss()
        bal = self.calculate_balance_loss()
        total = se + self.lambda_tv * tv + self.gamma_balance * bal
        return total, {
            'se': float(se.detach().cpu()),
            'tv': float(tv.detach().cpu()),
            'bal': float(bal.detach().cpu()),
            'lambda_tv': self.lambda_tv,
            'gamma_balance': self.gamma_balance,
            'balance_mode': self.balance_mode,
        }

    # ---------------- spectral pseudotime APIs -----------------------

    @torch.no_grad()
    def compute_spectral_tau(self, normalized: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        S = self.s_dic[self.height]
        A = self._A_for_traj
        return compute_tau_fiedler_from_SA(S, A, normalized=normalized, zscore=True)

    @torch.no_grad()
    def compute_spectral_pseudotime(
        self,
        normalized: bool = True,
        post_smooth: float = 0.0,
        iters: int = 5,
        keep_cluster_means: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full spectral pseudotime without orientation.
        Returns: (tau: (K,), t_node: (N,), A_c: (K,K))
        """
        S = self.s_dic[self.height]
        A = self._A_for_traj
        tau, A_c = compute_tau_fiedler_from_SA(S, A, normalized=normalized, zscore=True)
        t_node = node_pseudotime_from_S(S, tau, A_sparse=A,
                                        post_smooth=post_smooth,
                                        iters=iters,
                                        keep_cluster_means=keep_cluster_means)
        self.pseudotime_spectral = t_node
        return tau, t_node, A_c
