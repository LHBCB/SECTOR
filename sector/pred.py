# pred.py
import warnings
warnings.filterwarnings("ignore")

import os

import torch
import numpy as np
import pandas as pd

from .utility.metrics import cluster_metrics
from .utility.utils import (
    decoding_from_assignment,
    remove_small_islands_on_graph,
    plot_pseudotime_spatial,
    orient_pseudotime_by_spatial_anchor,
    orient_pseudotime_by_pred_cluster,
    plot_cluster_spatial
)
from .fit import FitResult  # dataclass from training module


# ---------------------------------------------------------------------
# Loading best / running forward
# ---------------------------------------------------------------------
def load_best_or_forward(args, model, model_path, dataset, device):
    """
    Load the saved best model if present (when args.save=True).
    Then run a forward pass to obtain s_dic and emb_dic.
    """
    if args.save and os.path.exists(model_path):
        # Backward-compatible load (weights_only isn't available in older torch)
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            s_dic, emb_dic, _ = model(dataset.adj, dataset.feature, dataset.degrees)
    else:
        with torch.no_grad():
            s_dic, emb_dic, _ = model(dataset.adj, dataset.feature, dataset.degrees)
    return s_dic, emb_dic


# ---------------------------------------------------------------------
# Post-hoc refinement + (optional) scoring
# ---------------------------------------------------------------------
def post_hoc_refine_and_score(dataset, model, pred_raw, args):
    """
    Remove small islands, attach back to adata, and (optionally) compute metrics.
    When args.eval_mode == 0, no label-based metrics are computed or stored.
    """
    # Read params from args (backward compatible)
    min_abs = int(getattr(args, 'island_min_abs', 20))
    min_frac = float(getattr(args, 'island_min_frac', 0.03))
    max_iter = int(getattr(args, 'island_max_iter', 2))

    pred_before = pred_raw.copy()
    pred_clean = remove_small_islands_on_graph(
        dataset.adj,
        pred_raw,
        min_abs=min_abs,
        min_frac=min_frac,
        max_iter=max_iter,
    )
    changed = (pred_before != pred_clean).sum()
    print(
        f"[Post hoc] Island cleaner changed {changed} spots "
        f"(min_abs={min_abs}, min_frac={min_frac}, max_iter={max_iter})"
    )

    dataset.adata.obs["pred_region"] = pred_clean

    # -------- optional evaluation (labels) --------
    eval_mode = int(getattr(args, "eval_mode", 1)) == 1
    if eval_mode:
        cm = cluster_metrics(dataset.labels, torch.as_tensor(pred_clean))
        expected_K, used_K, nmi, hom, com, ari, ami, _ = cm.evaluateFromLabel()
        print(
            f"[Post hoc] After island removal => "
            f"NMI={nmi:.6f}, HOM={hom:.6f}, COM={com:.6f}, "
            f"ARI={ari:.6f}, AMI={ami:.6f} "
            f"(K={used_K}/{expected_K})"
        )

        # Store final metrics on the AnnData object so they persist to disk
        metrics_dict = {
            "expected_K": int(expected_K),
            "used_K": int(used_K),
            "NMI": float(nmi),
            "HOM": float(hom),
            "COM": float(com),
            "ARI": float(ari),
            "AMI": float(ami),
        }
        dataset.adata.uns.setdefault("SECTOR", {})
        dataset.adata.uns["SECTOR"]["final_metrics"] = metrics_dict

        return nmi, ari

    # If not evaluating, don't compute/return label metrics
    return None, None


# ---------------------------------------------------------------------
# Run summary save (optional text log; off by default)
# ---------------------------------------------------------------------
def save_run_summary(args, nmi, ari, save_path):
    """Append run hyperparams and best metrics to the result file."""
    os.makedirs("output", exist_ok=True)
    with open(save_path, 'a') as f:
        f.write(
            f"lr={args.lr}, embed_dim={args.embed_dim}, k={args.k}, dropout={args.dropout}, "
            f"beta_f={args.beta_f}, epochs={args.epochs}, num_clusters={args.num_clusters}, "
            f"verbose={args.verbose}, activation={args.activation}, seed={args.seed} \n"
        )
        f.write(f"--------Best NMI: [{nmi}], Best ARI: [{ari}] \n")


# ---------------------------------------------------------------------
# Trajectory + exports
# ---------------------------------------------------------------------
def export_trajectory_and_embeddings(args, model, dataset, s_dic, emb_dic, dataset_name, slice_name):
    """
    Compute spectral pseudotime (oriented), save plot, embeddings, and .h5ad.
    """
    os.makedirs("output", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    with torch.no_grad():
        tau, t_node, A_c = model.compute_spectral_pseudotime(
            normalized=True,
            post_smooth=0.15,
            iters=10,
            keep_cluster_means=True
        )
        t_node = t_node.detach().cpu().numpy()
    
    # Orient by predicted clusters; let helper scale if requested.
    if getattr(args, "root_cluster", None):
        pred_region = dataset.adata.obs["pred_region"].values
        if args.root_cluster in pred_region:
            t_oriented = orient_pseudotime_by_pred_cluster(
                t_node, pred_region, root_cluster=args.root_cluster
            )
        else:
            raise ValueError(f"Root cluster not in predicted regions: {pred_region}")
    else:
        XY = dataset.adata.obsm["spatial"]
        t_oriented = orient_pseudotime_by_spatial_anchor(t_node, XY, anchor=args.spatial_anchor)

    dataset.adata.obs["pseudotime"] = pd.Series(t_oriented, index=dataset.adata.obs_names)

    if args.plot:
        plot_cluster_spatial(
            dataset.adata, 
            label=args.label,
            save_path=f"figures/{dataset_name}.{slice_name}.clusters.png",
            eval_mode=args.eval_mode
        )
        
        # Save a figure
        plot_pseudotime_spatial(
            dataset.adata,
            obs_key="pseudotime",
            save_path=f"figures/{dataset_name}.{slice_name}.pseudotime.png",
            title=""
        )

    # Save S/embeddings/A_c for downstream analysis
    S_used = s_dic[model.height].detach().cpu().numpy()
    sector_embedding = emb_dic[model.height].detach().cpu().numpy()
    dataset.adata.obsm["soft_ass_mat"] = S_used
    dataset.adata.obsm["sector_embedding"] = sector_embedding
    dataset.adata.uns["cluster_graph_Ac"] = A_c.detach().cpu().numpy()

    if args.save_adata:
        dataset.adata.write_h5ad(f"output/{dataset_name}.{slice_name}.sector.h5ad")


def infer_clusters_and_pseudotime(args, fit_result: FitResult):
    """
    Post-hoc island cleanup, optional metrics logging, and pseudotime export.
    Loads best checkpoint if saved, runs a forward pass to get s/embeddings,
    and writes outputs (figure + h5ad).
    Returns the AnnData with all new fields.
    """
    #os.makedirs("output", exist_ok=True)

    # load best or forward
    s_dic, emb_dic = load_best_or_forward(
        args=args,
        model=fit_result.model,
        model_path=fit_result.model_path,
        dataset=fit_result.dataset,
        device=fit_result.device,
    )

    # best hard labels (pre post-hoc)
    best_pred_raw = fit_result.best_state.best_pred_raw
    if best_pred_raw is None:
        with torch.no_grad():
            best_pred_raw = decoding_from_assignment(fit_result.model.hard_dic[1])
    if isinstance(best_pred_raw, torch.Tensor):
        best_pred_raw = best_pred_raw.detach().cpu().numpy().astype(int)

    # post-hoc (+ optional metrics)
    nmi, ari = post_hoc_refine_and_score(
        dataset=fit_result.dataset,
        model=fit_result.model,
        pred_raw=best_pred_raw,
        args=args,
    )

    # pseudotime + exports
    export_trajectory_and_embeddings(
        args=args,
        model=fit_result.model,
        dataset=fit_result.dataset,
        s_dic=s_dic,
        emb_dic=emb_dic,
        dataset_name=args.dataset,
        slice_name=args.slice,
    )

    return fit_result.dataset.adata
