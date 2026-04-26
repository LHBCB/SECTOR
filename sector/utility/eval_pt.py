from __future__ import annotations

from decimal import Decimal
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData

from scipy import sparse
from scipy.stats import spearmanr

from typing import Dict, Optional, Tuple
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import normalized_mutual_info_score

import re
import gseapy as gp
import os

import textwrap
from typing import Optional, Sequence, Dict, Iterable

# ----------------------------- Utilities -----------------------------

def compute_nmi(adata_out, pred_key, gt_key='Region'):

    mask = adata_out.obs[pred_key].notna() & adata_out.obs[gt_key].notna()
    y_true = adata_out.obs.loc[mask, gt_key].astype(str).values
    y_pred = adata_out.obs.loc[mask, pred_key].astype(str).values

    nmi = normalized_mutual_info_score(y_true, y_pred)
    return nmi

def _normalize_values(values: np.ndarray, how: str = "none") -> np.ndarray:
    v = np.asarray(values, float)
    if how == "none":
        return v
    if how == "zscore":
        m, s = np.nanmean(v), np.nanstd(v)
        return (v - m) / s if s > 0 else np.zeros_like(v)
    if how == "minmax":
        mn, mx = np.nanmin(v), np.nanmax(v)
        rng = mx - mn
        return (v - mn) / rng if rng > 0 else np.zeros_like(v)
    if how == "rank":
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v))
        return ranks / (len(v) - 1 if len(v) > 1 else 1.0)
    raise ValueError(f"Unknown normalization: {how!r}")

def _auto_lag_step(coords: np.ndarray, platform: Optional[str]) -> float:
    """
    Choose a lag step:
      - Visium: 167 pixels (the paper's setting).
      - MERFISH/other: median nearest-neighbor distance (≈ cell spacing; in the same units as coords).
    """
    if platform is not None and platform.lower() in {"visium", "10x", "10x visium"}:
        return 167.0  # pixels, as used in the stLearn paper
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)  # self + nearest neighbor
    nnd = dists[:, 1]
    return float(np.median(nnd))

def _prepare_bins(lag_step: float, max_range: Optional[float], n_bins: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if lag_step <= 0:
        raise ValueError("lag_step must be > 0")
    if n_bins is None and max_range is None:
        n_bins = 20
        max_range = lag_step * n_bins
    elif n_bins is None and max_range is not None:
        n_bins = max(1, int(np.floor(max_range / lag_step)))
        max_range = lag_step * n_bins
    elif n_bins is not None and max_range is None:
        n_bins = max(1, int(n_bins))
        max_range = lag_step * n_bins
    else:
        n_bins = max(1, int(np.floor(max_range / lag_step)))
        max_range = lag_step * n_bins

    edges = np.arange(0, n_bins + 1, dtype=float) * lag_step
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers

# ----------------------------- Result container -----------------------------

@dataclass
class VariogramResult:
    method: str
    h: np.ndarray               # bin centers (distance)
    gamma: np.ndarray           # semivariance per bin (normalized if selected)
    counts: np.ndarray          # number of pairs per bin
    mrs: float                  # Mean Relative Semivariance (lower is better)
    lag_step: float
    max_range: float
    platform: Optional[str]
    value_normalization: str
    normalized_by_variance: bool
    variance: float

# ----------------------------- Core variogram -----------------------------

def empirical_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    lag_step: Optional[float] = None,
    max_range: Optional[float] = None,
    n_bins: Optional[int] = None,
    min_pairs_per_bin: int = 30,
    normalize_by_variance: bool = True,
    platform: Optional[str] = None,
    value_normalization: str = "none",
    method_name: str = "method",
) -> VariogramResult:
    """
    Empirical variogram using the Matheron estimator, with optional variance-normalization.
    Returns bin centers h, semivariances γ(h), counts, and a single MRS summary.
    """
    coords = np.asarray(coords, float)
    values = _normalize_values(np.asarray(values, float), how=value_normalization)

    # Remove NaNs jointly
    mask = ~(np.isnan(values) | np.isnan(coords).any(axis=1))
    coords = coords[mask]
    values = values[mask]
    n = len(values)
    if n < 3:
        raise ValueError("Need at least 3 points to compute a variogram.")

    # Lag step and bins
    lag = _auto_lag_step(coords, platform) if lag_step is None else float(lag_step)
    edges, centers = _prepare_bins(lag, max_range, n_bins)
    max_range = edges[-1]

    # Pairwise distances and squared differences
    dists = pdist(coords, metric="euclidean")
    sqdiff = pdist(values.reshape(-1, 1), metric="euclidean") ** 2

    # Assign to bins [0, max_range]
    bin_idx = np.digitize(dists, edges, right=False) - 1  # 0..n_bins-1
    valid_pairs = (bin_idx >= 0) & (bin_idx < len(centers))
    bin_idx = bin_idx[valid_pairs]
    sqdiff = sqdiff[valid_pairs]

    # Aggregate: γ(h) = 0.5 * mean (difference^2)
    sums = np.bincount(bin_idx, weights=sqdiff, minlength=len(centers)).astype(float)
    counts = np.bincount(bin_idx, minlength=len(centers)).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = 0.5 * (sums / np.where(counts > 0, counts, np.nan))

    # Keep only sufficiently populated bins
    keep = (counts >= min_pairs_per_bin) & np.isfinite(gamma)
    h = centers[keep]
    gamma = gamma[keep]
    counts = counts[keep]

    # Variance-normalize (unitless variogram; sill ~ 1 if fully random)
    var = float(np.var(values, ddof=1))
    if normalize_by_variance and var > 0:
        gamma = gamma / var

    # MRS: unweighted average of retained γ(h); NaN if no usable bins
    mrs = float(np.nanmean(gamma)) if gamma.size > 0 else float("nan")

    return VariogramResult(
        method=method_name,
        h=h,
        gamma=gamma,
        counts=counts,
        mrs=mrs,
        lag_step=lag,
        max_range=max_range,
        platform=platform,
        value_normalization=value_normalization,
        normalized_by_variance=normalize_by_variance,
        variance=var,
    )

# ----------------------------- Batch comparison -----------------------------

def compare_pseudotime_methods(
    *,
    coords: np.ndarray,
    pseudotime_by_method: Dict[str, np.ndarray],
    platform: Optional[str] = None,
    lag_step: Optional[float] = None,
    max_range: Optional[float] = None,
    n_bins: Optional[int] = 20,
    value_normalization: str = "none",     # 'none' | 'zscore' | 'minmax' | 'rank'
    normalize_by_variance: bool = True,
    min_pairs_per_bin: int = 30,
):
    """
    Compute variograms and a single MRS for multiple pseudotime methods.
    Returns:
      - results: {method -> VariogramResult}
      - leaderboard: list of (method, MRS) sorted by ascending MRS (best first)
    """
    results = {}
    for name, vals in pseudotime_by_method.items():
        res = empirical_variogram(
            coords=coords,
            values=vals,
            lag_step=lag_step,
            max_range=max_range,
            n_bins=n_bins,
            min_pairs_per_bin=min_pairs_per_bin,
            normalize_by_variance=normalize_by_variance,
            platform=platform,
            value_normalization=value_normalization,
            method_name=name,
        )
        results[name] = res

    # Sort by MRS (lower is smoother). Treat NaN as worst.
    leaderboard = sorted(
        [(r.method, np.inf if np.isnan(r.mrs) else r.mrs) for r in results.values()],
        key=lambda t: t[1],
    )
    return results, leaderboard

# ----------------------------- Plotting helper ----------------------------
def plot_variograms(
    results: Dict[str, 'VariogramResult'],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_counts: bool = False,
    legend_loc: str = "outside right",
    ylabel: Optional[str] = None,
    method_colors: Optional[Dict[str, str]] = None,  # <--- NEW
    # Font sizes
    legend_title: str = "Method",           # NEW
    legend_title_fontsize: Optional[int] = None,  # NEW
    title_fontsize: int = 22,
    label_fontsize: int = 20,
    tick_fontsize: int = 20,
    legend_fontsize: int = 20,
    # NEW: thickness controls
    curve_lw: float = 3.0,          # variogram line width
    spine_lw: float = 2.5,          # surrounding box (spines)
    tick_width: float = 2.0,        # tick mark thickness
    tick_length: float = 6.0,       # tick length
    grid_lw: float = 1.5,           # grid line width
    draw_figure_frame: bool = False,# draw a border around the whole figure?
    figure_lw: float = 2.5,         # figure border thickness (if drawn)
    figure_edgecolor: str = "black" # figure border color
) -> plt.Axes:
    """
    Overlay variograms from several methods.
    If results were computed with normalize_by_variance=True, the y-axis is unitless
    (Relative/Normalized semivariance; sill ~ 1 for spatially uncorrelated fields).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    # Plot each variogram
    for name, res in results.items():
        x, y = res.h, res.gamma
        #lbl = f"{name} (MRS={res.mrs:.4f})" if np.isfinite(res.mrs) else f"{name} (MRS=NA)"
        lbl = name
        
        # pick color from dict if available
        color = None
        if method_colors is not None:
            color = method_colors.get(name, None)

        ax.plot(x, y, lw=curve_lw, label=lbl, color=color)
        
        if show_counts:
            s = 10 * (res.counts / (res.counts.max() if res.counts.max() > 0 else 1))
            ax.scatter(x, y, s=s, alpha=0.35)

    first = next(iter(results.values()))
    ax.set_xlabel("Distance lag", fontsize=label_fontsize)

    if ylabel is None:
        ax.set_ylabel(
            "Semivariance"
            if first.normalized_by_variance
            else "Semivariance γ(h) (Matheron)",
            fontsize=label_fontsize,
        )
    else:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    # Grid thicker
    #ax.grid(True, alpha=0.25, linewidth=grid_lw)
    ax.grid(False, which="both", axis="x") 

    # Thicken the surrounding box (spines)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)

    # Thicken tick marks and set tick label font size
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=tick_width, length=tick_length)
    
    # Legend
    #ax.legend(loc=legend_loc, frameon=False, fontsize=legend_fontsize)
    # ------- Legend with title & adjustable title fontsize -------
    if legend_loc == "outside right":
        legend = ax.legend(
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
            fontsize=legend_fontsize,
        )
    else:
        legend = ax.legend(
            title=legend_title,
            loc=legend_loc,
            frameon=False,
            fontsize=legend_fontsize,
        )

    # If not specified, default legend title size to legend_fontsize
    if legend_title_fontsize is None:
        legend_title_fontsize = legend_fontsize
    legend.get_title().set_fontsize(legend_title_fontsize)
    # -------------------------------------------------------------
    
    # Optional: draw a visible border around the entire figure (off by default)
    if draw_figure_frame:
        fig.patch.set_edgecolor(figure_edgecolor)
        fig.patch.set_linewidth(figure_lw)

    return ax

def _bh_fdr_archive(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction; preserves NaNs."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not np.any(ok):
        return q
    p_ok = p[ok]
    m = p_ok.size
    order = np.argsort(p_ok)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q_ok = p_ok * m / ranks
    q_ok = np.minimum.accumulate(q_ok[order[::-1]])[::-1]
    q_ok = np.clip(q_ok, 0.0, 1.0)
    q[ok] = q_ok
    return q

def _bh_fdr(p: np.ndarray) -> np.ndarray:
    reject, pvals_bh, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    #print("Reject at q=0.05? ", reject)
    #print("BH-adjusted p-values:", pvals_bh)
    return pvals_bh

def _get_matrix_and_genes(
    adata: AnnData,
    *,
    use_raw: bool = False,
    layer: Optional[str] = None,
) -> tuple[np.ndarray | sparse.spmatrix, Iterable[str]]:
    """Return (X, var_names) honoring raw/layer."""
    if use_raw and adata.raw is not None:
        X = adata.raw.X
        genes = adata.raw.var_names
    elif layer is not None:
        X = adata.layers[layer]
        genes = adata.var_names
    else:
        X = adata.X
        genes = adata.var_names
    return X, np.asarray(genes)


def compute_pseudotime_gene_correlations(
    adata: AnnData,
    *,
    pseudotime_key: str = "pseudotime",
    use_raw: bool = False,
    layer: Optional[str] = None,
    min_detected_fraction: float = 0.0,
    min_variance: float = 0.0,
) -> pd.DataFrame:
    """
    Compute Spearman correlation for each gene vs GLOBAL pseudotime in adata.obs[pseudotime_key].

    Returns a DataFrame with index=gene and columns: rho, pval, qval (BH-FDR), n_used.
    Filters genes with low detection (min_detected_fraction) and zero/low variance (min_variance).
    """
    if pseudotime_key not in adata.obs:
        raise KeyError(f"'{pseudotime_key}' not found in adata.obs")

    # Get matrix and genes
    X, genes = _get_matrix_and_genes(adata, use_raw=use_raw, layer=layer)

    # Keep only observations with finite pseudotime
    pt = np.asarray(adata.obs[pseudotime_key]).astype(float)
    ok_obs = np.isfinite(pt)
    if not np.any(ok_obs):
        raise ValueError("All pseudotime values are NaN/inf.")
    pt = pt[ok_obs]

    X = X[ok_obs, :]
    is_sparse = sparse.issparse(X)

    n_obs = X.shape[0]
    n_genes = X.shape[1]

    # Simple filters (optional, cheap)
    if is_sparse:
        detected = np.asarray((X > 0).sum(axis=0)).ravel() / n_obs
        mean = np.asarray(X.mean(axis=0)).ravel()
        sqmean = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = sqmean - mean**2
    else:
        detected = (X > 0).mean(axis=0)
        var = np.var(X, axis=0)

    keep = (detected >= float(min_detected_fraction)) & (var > float(min_variance))
    if not np.any(keep):
        raise ValueError("No genes pass the detection/variance filters.")

    # Allocate outputs
    rhos = np.full(n_genes, np.nan, dtype=float)
    pvals = np.full(n_genes, np.nan, dtype=float)
    n_used = np.zeros(n_genes, dtype=int)

    # Compute Spearman one gene at a time (robust and clear)
    # (Vectorized rank-based version is possible, but this is easiest to read/maintain.)
    for j in np.where(keep)[0]:
        if is_sparse:
            x = X[:, j].toarray().ravel()
        else:
            x = np.asarray(X[:, j]).ravel()

        # Drop NaNs if any
        ok = np.isfinite(x)
        if ok.sum() < 3:
            continue
        r, p = spearmanr(pt[ok], x[ok])
        rhos[j] = r
        pvals[j] = p
        n_used[j] = ok.sum()

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "gene": genes,
            "rho": rhos,
            "pval": pvals,
            "qval": _bh_fdr(pvals),
            "n_used": n_used,
            "detected_fraction": detected,
            "variance": var,
        }
    ).set_index("gene").sort_values("rho", ascending=False)

    # Store in .uns for convenience
    adata.uns["pseudotime_markers_global"] = df
    return df
    
def transition_markers_plot_global(
    adata: AnnData,
    *,
    top_genes: int = 10,
    contrast_colors=["#fb687a", "#31a2fb"],
    pseudotime_key: str = "pseudotime",
    use_raw: bool = False,
    layer: Optional[str] = None,
    dpi: int = 150,
    output: Optional[str] = None,
    name: Optional[str] = None,
    show_pvalues: bool = True,
    spine_lw: float = 2.0,
    corr_threshold: float = 0.3,   # only show genes with |rho| > this
    label_fontsize: int = 10,      # <- NEW: axis label font size
    tick_fontsize: int = 10,       # <- NEW: tick label font size
) -> AnnData:
    """
    Compute gene correlations vs GLOBAL pseudotime and plot top up-/down-markers
    in a mirrored horizontal bar plot. Only genes with |rho| > corr_threshold
    are visualised. Results table is stored at adata.uns['pseudotime_markers_global'].
    """
    # Compute/fetch correlations
    df = compute_pseudotime_gene_correlations(
        adata,
        pseudotime_key=pseudotime_key,
        use_raw=use_raw,
        layer=layer,
    )
    df = df[np.isfinite(df["rho"])].copy()
    adata.uns["pseudotime_markers_global"] = df

    # Apply threshold and take top genes
    pos = (
        df[df["rho"] > corr_threshold]
        .sort_values("rho", ascending=False)
        .head(top_genes)
        .copy()
    )
    neg = (
        df[df["rho"] < -corr_threshold]
        .sort_values("rho", ascending=True)
        .head(top_genes)
        .copy()
    )

    # Arrays for plotting (reverse so strongest near center)
    x_pos = list(pos["rho"])[::-1]
    x_neg = list(neg["rho"])[::-1]
    y_pos = np.arange(len(x_pos))
    y_neg = np.arange(len(x_neg))
    max_len = max(len(x_pos), len(x_neg), 1)

    # Figure
    fig, axes = plt.subplots(ncols=2, sharey=True, dpi=dpi, figsize=(6.4, 3.8))
    axes[0].barh(y_neg, x_neg, align="center", color=contrast_colors[0])  # soft teal/green
    axes[1].barh(y_pos, x_pos, align="center", color=contrast_colors[1])  # soft peach/orange
    fig.subplots_adjust(wspace=0)
    axes[0].set_ylim(-0.5, max_len - 0.5)
    axes[1].set_ylim(-0.5, max_len - 0.5)

    # Cosmetics
    for ax in axes:
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(spine_lw)          # apply spine thickness

        ax.get_yaxis().set_ticks([])

        # <- increase tick *label* font size (x only; y has no ticks)
        ax.tick_params(axis="x", which="both", labelsize=tick_fontsize, length=0)

    # Annotate positives
    rects = axes[1].patches
    pos_names = list(pos.index)[::-1]
    pos_p = list(pos["pval"])[::-1] if "pval" in pos else [np.nan]*len(pos)
    for i, rect in enumerate(rects):
        gene_name = pos_names[i]
        p_text = (
            "{:.2E}".format(Decimal(str(pos_p[i])))
            if (show_pvalues and np.isfinite(pos_p[i]))
            else ""
        )
        axes[1].text(rect.get_x() + rect.get_width() + 0.01,
                     rect.get_y() + rect.get_height() / 2.0,
                     gene_name, ha="left", va="center", size=7)
        if p_text:
            axes[1].text(rect.get_x() + 0.01,
                         rect.get_y() + rect.get_height() / 2.0,
                         p_text, color="w", ha="left", va="center", size=7)

    # Annotate negatives
    rects = axes[0].patches
    neg_names = list(neg.index)[::-1]
    neg_p = list(neg["pval"])[::-1] if "pval" in neg else [np.nan]*len(neg)
    for i, rect in enumerate(rects):
        gene_name = neg_names[i]
        p_text = (
            "{:.2E}".format(Decimal(str(neg_p[i])))
            if (show_pvalues and np.isfinite(neg_p[i]))
            else ""
        )
        axes[0].text(rect.get_x() + rect.get_width() - 0.01,
                     rect.get_y() + rect.get_height() / 2.0,
                     gene_name, ha="right", va="center", size=7)
        if p_text:
            axes[0].text(rect.get_x() - 0.01,
                         rect.get_y() + rect.get_height() / 2.0,
                         p_text, color="w", ha="right", va="center", size=7)

    # Single shared x-axis label (no title)
    try:
        fig.supxlabel("Spearman correlation coefficient", fontsize=label_fontsize)
    except Exception:
        fig.text(0.5, 0.02, "Spearman correlation coefficient",
                 ha="center", va="center", fontsize=label_fontsize)

    # Save if requested
    if name is None:
        name = "pseudotime_markers_global.png"
    if output is not None and name is not None:
        os.makedirs('enrich', exist_ok=True)
        fig.savefig(f"{output}/{name}", dpi=dpi, bbox_inches="tight", pad_inches=0)

    plt.show()
    return adata

def get_top_genes(
    adata,
    method,
    pseudotime_key,
    *,
    top_genes=20,
    n_top_hvgs=2000,
    hvg_flavor="seurat_v3",
    hvg_layer="counts",
    use_hvg=True,
):
    """
    By default, run pseudotime-gene correlation on HVGs only.

    Priority:
    1) If adata.var['highly_variable'] exists and contains True values, use those HVGs.
    2) Otherwise compute HVGs first (default 2000), then subset to them.
    3) If use_hvg=False, use all genes.
    """

    # work on a copy so the input adata is not modified in place
    adata_use = adata.copy()

    if use_hvg:
        has_hvg_col = "highly_variable" in adata_use.var.columns
        has_any_hvg = False

        if has_hvg_col:
            hvg_mask = adata_use.var["highly_variable"].fillna(False).astype(bool).to_numpy()
            has_any_hvg = hvg_mask.any()

        if has_hvg_col and has_any_hvg:
            adata_use = adata_use[:, hvg_mask].copy()
            print(f"Using existing HVGs in adata.var['highly_variable']: {adata_use.n_vars} genes")
        else:
            hvg_kwargs = dict(
                flavor=hvg_flavor,
                n_top_genes=n_top_hvgs,
                subset=True,
            )

            # use layer='counts' when available, as requested
            if hvg_layer is not None and hvg_layer in adata_use.layers:
                hvg_kwargs["layer"] = hvg_layer

            sc.pp.highly_variable_genes(adata_use, **hvg_kwargs)
            print(f"'highly_variable' not found (or empty); computed HVGs and subset to {adata_use.n_vars} genes")
    else:
        print(f"Using all genes: {adata_use.n_vars} genes")

    os.makedirs("./enrich", exist_ok=True)

    transition_markers_plot_global(
        adata_use,
        top_genes=top_genes,
        pseudotime_key=pseudotime_key,
        use_raw=False,
        layer=None,
        dpi=150,
        output="./enrich",
        name=f"{method}.global_pseudotime_markers.png",
    )

    df = adata_use.uns["pseudotime_markers_global"].copy()

    top_pos_genes = df[(df["rho"] > 0.3) & (df["qval"] < 0.05)].index[:100].tolist()
    print(f"Found {len(top_pos_genes)} top up-regulated transition genes: ", top_pos_genes)

    top_neg_genes = df[(df["rho"] < -0.3) & (df["qval"] < 0.05)].index[:100].tolist()
    print(f"Found {len(top_neg_genes)} top down-regulated transition genes: ", top_neg_genes)

    return top_pos_genes, top_neg_genes

def choose_libraries(preferred_patterns):
    avail = gp.get_library_name()
    chosen = []
    for pat in preferred_patterns:
        r = re.compile(pat, flags=re.I)
        candidates = [lib for lib in avail if r.search(lib)]
        if candidates:
            # choose the last after natural sort = usually the most recent version
            chosen.append(sorted(candidates)[-1])
    return list(dict.fromkeys(chosen))  # unique, keep order

#Run Enrichr ORA for both directions across selected libraries
def run_enrichr_lists(gene_list, libraries, *, organism='Human', label='up', cutoff=0.05):
    rows = []
    for lib in libraries:
        enr = gp.enrichr(gene_list=gene_list, gene_sets=lib, organism=organism, cutoff=cutoff, outdir=None, no_plot=True)
        if enr.results is None or enr.results.empty:
            continue
        res = enr.results.copy()
        res['library'] = lib
        res['direction'] = label
        rows.append(res)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def get_enrichment(top_pos_genes, top_neg_genes, libs, method):
    res_up   = run_enrichr_lists(top_pos_genes, libs, label='up')
    res_down = run_enrichr_lists(top_neg_genes, libs, label='down')
    res_all  = pd.concat([res_up, res_down], ignore_index=True)
    
    # Keep key columns; sort by adjusted p-value
    keep_cols = ['Term','Adjusted P-value','Odds Ratio','Combined Score','library','direction','Overlap','P-value','Genes']
    res_all = res_all[keep_cols].sort_values(['library','direction','Adjusted P-value'])
    res_all = res_all[res_all['Adjusted P-value']<0.05].copy()
    os.makedirs('enrich', exist_ok=True)
    res_all.to_csv(f'enrich/{method}_enrich_full.csv', header=True, index=False)
    return res_all

def plot_top_enrichment_bars(
    df: pd.DataFrame,
    *,
    top_k: int = 15,
    term_col: Optional[str] = None,
    padj_col: Optional[str] = None,
    wrap: int = 60,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    output: Optional[str] = None,
    # --- new font knobs ---
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    spine_lw: float = 2.5, 
) -> plt.Axes:
    """
    Make a horizontal bar chart of the top-k enrichment terms
    using -log10(adjusted p-value) as bar length.
    """

    # 1) Auto-detect columns if not provided
    if term_col is None:
        candidates_terms: Sequence[str] = (
            "term", "name", "description", "Term", "Pathway", "Gene_set", "Description"
        )
        for c in candidates_terms:
            if c in df.columns:
                term_col = c
                break
        if term_col is None:
            raise ValueError("Could not find a term/description column. "
                             "Pass `term_col=...` explicitly.")

    if padj_col is None:
        candidates_padj: Sequence[str] = (
            "padj", "p_adj", "p_adjust", "p_adjusted", "adj_pval",
            "Adjusted P-value", "Adjusted P-value (FDR)", "qvalue", "q_value", "FDR"
        )
        for c in candidates_padj:
            if c in df.columns:
                padj_col = c
                break
        if padj_col is None:
            raise ValueError("Could not find an adjusted p-value column. "
                             "Pass `padj_col=...` explicitly.")

    work = df[[term_col, padj_col]].copy()
    work = work.dropna(subset=[padj_col])
    if work.empty:
        raise ValueError("No rows with non-null adjusted p-values.")

    # 2) Compute -log10(padj); guard against zeros
    eps = np.finfo(float).tiny
    work["neglog10_p"] = -np.log10(np.maximum(work[padj_col].astype(float), eps))

    # 3) Select top_k by padj and order for plotting (most significant on top)
    work = work.sort_values(padj_col, ascending=True).head(top_k)
    work = work.iloc[::-1]

    # 4) Wrap long labels
    def _wrap(s: str) -> str:
        return textwrap.fill(str(s), width=wrap)
    y_labels = work[term_col].map(_wrap).tolist()

    # 5) Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y_labels, work["neglog10_p"].values, color='plum')

    ax.set_xlabel(r"$-\log_{10}(\mathrm{adjusted\ p\text{-}value})$", fontsize=label_fontsize)
    ax.set_ylabel("", fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)

    # ---- Keep only the left (y) and bottom (x) axis lines (hide the surrounding box)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    # Ensure ticks only on the visible axes
    ax.tick_params(top=False, right=False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)
    
    # Tick label font size (both axes)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    plt.tight_layout()

    if output is not None:
        fig.savefig('enrich/' + output, dpi=300, bbox_inches="tight")

    return ax

def plot_pseudotime(adata, obs_key="pseudotime", cmap="viridis",
                            s=None, alpha=1.0, invert_y=True, title=None,
                            save_path=None, dpi=200, ax=None,                             
                            cbar_title=None,              # NEW: text
                            cbar_title_at_top=False,      # NEW: place above instead of along bar
                            cbar_labelpad=-5,              # NEW: spacing
                            cbar_fontweight="normal",
                            cbar_labelsize=25, cbar_ticksize=20):

    if "spatial" not in adata.obsm_keys():
        raise ValueError("adata.obsm['spatial'] not found.")
    coords = np.asarray(adata.obsm["spatial"])
    t = np.asarray(adata.obs[obs_key].values, dtype=float)
    N = coords.shape[0]
    s = s if s is not None else max(1.0, 20000.0 / max(1, N))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

    pts = ax.scatter(coords[:, 0], coords[:, 1], c=t, s=s, alpha=alpha,
                     cmap=cmap, edgecolors='none')
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_axis_off()

    cbar = plt.colorbar(pts, ax=ax, fraction=0.046, pad=0.04)
    # only show 0 and 1.0
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["0", "1"])

    #cbar.set_label(obs_key, fontsize=cbar_labelsize)   # label font size

    text = cbar_title or obs_key
    if cbar_title_at_top:
        cbar.ax.set_title(text, fontsize=cbar_labelsize,
                          pad=cbar_labelpad, fontweight=cbar_fontweight)
    else:
        cbar.set_label(text, fontsize=cbar_labelsize,
                       labelpad=cbar_labelpad, fontweight=cbar_fontweight)
    
    cbar.ax.tick_params(labelsize=cbar_ticksize)       # tick font size

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    return ax
    
def _plot_spatial_column(
    adata,
    col,
    *,
    palette=None,
    ax=None,
    plot_title=None,          # None means no plot title
    legend_title=None,        # will become legend title
    legend_loc="right margin",
    use_uns_colors=True,
    point_size=20,
    legend_fontsize=8,
    legend_markerscale=5.0
):
    import math
    import matplotlib.pyplot as plt
    import scanpy as sc

    if col not in adata.obs:
        raise KeyError(f"'{col}' not found in adata.obs.")
    adata.obs[col] = adata.obs[col].astype("category")

    # set up coords
    if "spatial" in adata.obsm and adata.obsm["spatial"] is not None:
        adata.obs["x_coord"] = adata.obsm["spatial"][:, 0]
        adata.obs["y_coord"] = adata.obsm["spatial"][:, 1]
    else:
        if not {"x_coord", "y_coord"}.issubset(adata.obs.columns):
            raise KeyError("Missing spatial coords.")

    cats = list(adata.obs[col].cat.categories)

    def _palette_from_uns():
        if not use_uns_colors:
            return None
        pal = adata.uns.get(f"{col}_colors", None)
        if pal is None:
            return None
        if isinstance(pal, dict):
            return [pal.get(c, "#808080") for c in cats]
        pal = list(pal)
        if len(pal) < len(cats):
            pal += ["#808080"] * (len(cats) - len(pal))
        elif len(pal) > len(cats):
            pal = pal[:len(cats)]
        return pal

    if palette is None:
        final_palette = _palette_from_uns()
    else:
        pal = palette
        if isinstance(pal, dict):
            final_palette = [pal.get(c, "#808080") for c in cats]
        else:
            pal = list(pal)
            if len(pal) < len(cats):
                pal += ["#808080"] * (len(cats) - len(pal))
            elif len(pal) > len(cats):
                pal = pal[:len(cats)]
            final_palette = pal

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 7))

    # THIS is where point_size is used
    sc.pl.scatter(
        adata,
        x="x_coord",
        y="y_coord",
        color=col,
        size=point_size,          # <-- spatial dot size
        frameon=False,
        ax=ax,
        show=False,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        palette=final_palette,
    )

    if plot_title:
        ax.set_title(plot_title, size=30)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()

    # ---- move legend to bottom & multi-column, WITHOUT touching sizes ----
    if legend_title is None:
        legend_title = col

    old_leg = ax.get_legend()
    if old_leg is not None:
        handles, labels = ax.get_legend_handles_labels()
        old_leg.remove()

        n_items = len(labels)
        max_rows = 6
        ncol = max(1, math.ceil(n_items / max_rows))

        legend_markerscale = legend_markerscale  # legend dot size multiplier

        leg = ax.legend(
            handles,
            labels,
            title=legend_title,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            frameon=False,
            fontsize=legend_fontsize,
            markerscale=legend_markerscale,  # only affects legend markers
        )
        if legend_fontsize is not None:
            leg.get_title().set_fontsize(legend_fontsize)

    return fig, ax


def plot_pred_region(
    adata,
    *,
    pred_key="pred_region",
    palette=None,
    legend_title="Domains",     # <- new
    save_path=None,
    use_uns_colors=True,
    point_size=20,
    legend_loc="right margin",
    legend_fontsize=20,
    legend_markerscale=5.0,
    plot_title=None,
    ax=None,
    show=True,
    close=True,
):
    import os
    import matplotlib.pyplot as plt

    fig, ax = _plot_spatial_column(
        adata,
        pred_key,
        palette=palette,
        ax=ax,
        plot_title=plot_title,              # <- removes axes title
        legend_title=legend_title,    # <- puts title in legend
        legend_loc=legend_loc,
        use_uns_colors=use_uns_colors,
        point_size=point_size,
        legend_fontsize=legend_fontsize,
        legend_markerscale=legend_markerscale
    )

    plt.tight_layout(pad=3.0)
    
    if not plot_title:
        ax.set_title("")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        (fig or ax.figure).savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if close and fig is not None:
        plt.close(fig)
    return fig, ax

def plot_region(
    adata,
    *,
    label="Region",
    legend_title='Annotation',
    palette=None,          # <- optional
    save_path=None,
    use_uns_colors=True,
    point_size=10,
    legend_loc="right margin",
    legend_fontsize=18,
    legend_markerscale=5.0,
    plot_title=None,
    ax=None,
    show=True,
    close=True,
):
    import os
    import matplotlib.pyplot as plt

    fig, ax = _plot_spatial_column(
        adata,
        label,
        palette=palette,
        ax=ax,
        plot_title=plot_title,              # <- removes axes title
        legend_title=legend_title,    # <- puts title in legend
        legend_loc=legend_loc,
        use_uns_colors=use_uns_colors,
        point_size=point_size,
        legend_fontsize=legend_fontsize,
        legend_markerscale=legend_markerscale
    )

    if not plot_title:
        ax.set_title("")
    
    plt.tight_layout(pad=3.0)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        (fig or ax.figure).savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if close and fig is not None:
        plt.close(fig)
    return fig, ax