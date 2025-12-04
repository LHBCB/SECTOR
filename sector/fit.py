import warnings
warnings.filterwarnings("ignore")

import os
import time
from collections import deque
from dataclasses import dataclass

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi_score

from .utility.metrics import cluster_metrics
from .utility.parser import parse_args
from .utility.dataset import STData
from .utility.utils import (
    set_seed,
    set_tv_weight_on_model,
    set_balance,
    warmup_factor_epoch,
    decoding_from_assignment,
    compute_se_spatial,
    compute_edge_agreement_scores,
)
from .model import SEModel

torch.autograd.set_detect_anomaly(True)
#print(torch.cuda.is_available())

# ---------------------------------------------------------------------
# Early-stopping configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class EarlyStoppingConfig:
    unsup_patience_checks: int = 6      # #verbose-checks with no improvement
    rel_improve_tol: float = 0.005      # relative improvement tolerance
    stability_nmi_thr: float = 0.99     # assignment self-NMI threshold
    stability_usedk_window: int = 4     # UsedK must be stable over this window
    stability_hits_required: int = 3    # consecutive stable hits required


def cfg_from_args(args) -> EarlyStoppingConfig:
    """Build early-stopping config, allowing overrides via args."""
    return EarlyStoppingConfig(
        unsup_patience_checks=getattr(args, 'unsup_patience_checks', 6),
        rel_improve_tol=getattr(args, 'rel_improve_tol', 0.005),
        stability_nmi_thr=getattr(args, 'stability_nmi_thr', 0.99),
        stability_usedk_window=getattr(args, 'stability_usedk_window', 4),
        stability_hits_required=getattr(args, 'stability_hits_required', 3),
    )


# ---------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------
class BestState:
    """Container for best label-free state during training."""
    def __init__(self):
        self.best_se_spatial = float('inf')
        self.best_eas_soft = -1.0
        self.best_unsup_epoch = None
        self.best_pred_raw = None
        self.best_nmi = None  # only set/printed when args.eval_mode == 1


@dataclass
class FitResult:
    """Return object from fit() with everything pred() needs."""
    model: SEModel
    dataset: STData
    device: str
    model_path: str
    best_state: BestState


# ---------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------
def setup_device_and_data(args):
    """Initialize device and dataset; move tensors to device and print stats."""
    device = f'cuda:{args.gpu}' if (args.gpu >= 0 and torch.cuda.is_available()) else 'cpu'
    print('Device:', device)

    dataset = STData(args, device)
    dataset.adj = dataset.adj.to(device)
    dataset.feature = dataset.feature.to(device)
    dataset.print_statistic()
    #print(dataset.adj._nnz(), dataset.weight.min().item(), dataset.weight.max().item())

    model_path = f'./sector_model/{args.dataset}_{args.slice}_K{args.num_clusters}.pt'
    return device, dataset, model_path


def setup_tv_warmup(args):
    """TV warmup bookkeeping."""
    lambda_tv_target = args.lambda_tv
    start_epoch = 0
    tv_warmup_epochs = args.tv_warmup_epochs
    end_epoch = start_epoch + tv_warmup_epochs
    return lambda_tv_target, start_epoch, end_epoch


def init_model_and_optimizer(args, dataset, device):
    """Create model + Adam optimizer; start TV weight at 0."""
    model = SEModel(args, dataset.feature, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    set_tv_weight_on_model(model, 0.0)
    return model, optimizer


# ---------------------------------------------------------------------
# Balance probe (optional)
# ---------------------------------------------------------------------
def maybe_run_balance_probe(args, model, optimizer, dataset, device,
                            lambda_tv_target, start_epoch, end_epoch):
    """
    Optional probe to see if the model naturally uses all K clusters
    before enabling balance prior. May restart model if probe fails.
    """
    probe_epochs = args.balance_probe_epochs
    stability = int(0.1 * probe_epochs)
    use_probe = (probe_epochs > 0 and args.gamma_balance > 0.0)
    restart_epoch = 0

    if not use_probe:
        return model, optimizer, restart_epoch

    eval_mode = int(getattr(args, "eval_mode", 1)) == 1
    set_balance(model, args, False)
    print(f"[Probe] Starting {probe_epochs} epochs with balance OFF...")
    stable_hits = 0

    for p_epoch in range(probe_epochs):
        tv_scale = warmup_factor_epoch(p_epoch, start_epoch, end_epoch)
        cur_lambda_tv = lambda_tv_target * tv_scale
        set_tv_weight_on_model(model, cur_lambda_tv)

        loss = forward_train_step(model, dataset, optimizer)

        hard_eval = model.hard_dic[1]
        used_K = int((hard_eval.sum(dim=0) > 0).sum().item())
        expected_K = hard_eval.shape[1]

        if (p_epoch + 1) > (probe_epochs - stability):
            if used_K == expected_K:
                stable_hits += 1
            else:
                stable_hits = 0

        if (p_epoch % max(10, getattr(args, 'verbose', 20)) == 0) or (p_epoch + 1 == probe_epochs):
            if eval_mode:
                print(f"[Probe] Epoch {p_epoch} | Loss={loss:.6f} | UsedK={used_K}/{expected_K}")
            else:
                print(f"[Probe] Epoch {p_epoch} | Loss={loss:.6f}")

    if stable_hits < stability:
        print(f"[Restart] Not stably at K by end of probe (stable_hits={stable_hits}/{stability}). "
              f"Restarting with gamma_balance={getattr(args, 'gamma_balance', 0.0)}.")
        del model, optimizer
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        set_seed(args)
        model = SEModel(args, dataset.feature, device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        set_balance(model, args, True)
        set_tv_weight_on_model(model, 0.0)
        restart_epoch = start_epoch
    else:
        print(f"[Continue] K stable at end of probe. Continuing with balance OFF.")
        set_balance(model, args, False)
        restart_epoch = start_epoch + probe_epochs

    return model, optimizer, restart_epoch


# ---------------------------------------------------------------------
# Core step helpers
# ---------------------------------------------------------------------
def forward_train_step(model, dataset, optimizer):
    """One optimization step (forward -> loss -> backward -> update)."""
    model.train()
    model(dataset.adj, dataset.feature, dataset.degrees)
    loss, _ = model.total_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_current_assignments(model, dataset):
    """
    Decode hard assignments and compute label-based metrics (logging only).
    Returns pred_tensor, pred_np, stats (expected_K, used_K, nmi, hom, com, ari, ami)
    """
    pred_tensor = decoding_from_assignment(model.hard_dic[1])
    cm = cluster_metrics(dataset.labels, pred_tensor)
    expected_K, used_K, nmi, hom, com, ari, ami, _ = cm.evaluateFromLabel()

    pred_np = pred_tensor.detach().cpu().numpy()
    stats = {
        "expected_K": expected_K,
        "used_K": used_K,
        "nmi": nmi,
        "hom": hom,
        "com": com,
        "ari": ari,
        "ami": ami,
    }
    return pred_tensor, pred_np, stats


def compute_unsup_diagnostics(model, dataset, hard_pred):
    """
    Compute label-free diagnostics on the spatial graph:
    - SE_spatial
    - Edge agreement (soft & hard)
    - Used-K
    """
    S_last = model.s_dic[model.height]  # (N, K)

    # ensure hard_pred is on the same device as S_last / edge_index
    if hard_pred is not None and isinstance(hard_pred, torch.Tensor):
        hard_pred = hard_pred.to(S_last.device)

    se_spat = compute_se_spatial(S_last, dataset.edge_index, dataset.weight)
    eas_soft, eas_hard = compute_edge_agreement_scores(
        S_last, dataset.edge_index, dataset.weight, hard_pred=hard_pred
    )

    used_K = int((model.hard_dic[1].sum(dim=0) > 0).sum().item())
    return se_spat, eas_soft, eas_hard, used_K


# ---------------------------------------------------------------------
# Training loop + early stopping
# ---------------------------------------------------------------------
def run_training_loop(args, model, optimizer, dataset,
                      lambda_tv_target, start_epoch, end_epoch,
                      restart_epoch, model_path, device,
                      early_cfg: EarlyStoppingConfig):
    """
    Full training loop with label-free diagnostics, stability checks,
    early stopping, and optional model checkpointing.
    """
    best = BestState()

    usedk_hist = deque(maxlen=early_cfg.stability_usedk_window)
    stability_hits = 0
    unsup_noimpr_cnt = 0
    last_pred_for_stability = None
    eval_mode = int(getattr(args, "eval_mode", 1)) == 1

    for epoch in range(restart_epoch, args.epochs):
        t0 = time.time()

        # TV warmup
        tv_scale = warmup_factor_epoch(epoch, start_epoch, end_epoch)
        cur_lambda_tv = lambda_tv_target * tv_scale
        set_tv_weight_on_model(model, cur_lambda_tv)

        # 1) optimize one step
        loss = forward_train_step(model, dataset, optimizer)

        # 2) predictions (+ optional label-based metrics)
        pred_tensor = decoding_from_assignment(model.hard_dic[1])
        pred_np = pred_tensor.detach().cpu().numpy()

        if eval_mode:
            _, _, stats = evaluate_current_assignments(model=model, dataset=dataset)
            print(
                f"Epoch: {epoch} [{time.time()-t0:.3f}s], "
                f"Loss: {loss:.6f}, "
                f"NMI: {stats['nmi']:.6f}, HOM: {stats['hom']:.6f}, COM: {stats['com']:.6f}, "
                f"K: {stats['used_K']}/{stats['expected_K']}, "
            )
        else:
            # Minimal logging when labels are not used
            print(
                f"Epoch: {epoch} [{time.time()-t0:.3f}s], "
                f"Loss: {loss:.6f}"
            )

        # 3) label-free diagnostics for early stopping
        se_spat, eas_soft, eas_hard, used_K = compute_unsup_diagnostics(
            model=model, dataset=dataset, hard_pred=pred_tensor
        )

        # UsedK stability window
        usedk_hist.append(used_K)
        usedk_stable = (len(usedk_hist) == usedk_hist.maxlen) and (len(set(list(usedk_hist))) == 1)

        # Self-NMI vs previous assignment (label-free)
        assign_nmi = None
        if last_pred_for_stability is not None:
            assign_nmi = nmi_score(last_pred_for_stability, pred_np)
        last_pred_for_stability = pred_np

        # 4) track improvements and optionally save
        improved = False
        if se_spat < (best.best_se_spatial * (1.0 - early_cfg.rel_improve_tol)):
            best.best_se_spatial = se_spat
            best.best_unsup_epoch = epoch
            best.best_pred_raw = pred_tensor
            if eval_mode and '_' in locals():
                # keep best NMI only for informative logging
                best.best_nmi = stats['nmi']
            unsup_noimpr_cnt = 0
            improved = True
            if args.save:
                os.makedirs("sector_model", exist_ok=True)
                torch.save(model.state_dict(), model_path)

        if eas_soft > (best.best_eas_soft * (1.0 + early_cfg.rel_improve_tol)):
            best.best_eas_soft = eas_soft
            unsup_noimpr_cnt = 0
            improved = True

        if not improved:
            unsup_noimpr_cnt += 1

        # 5) stability counter
        if (assign_nmi is not None) and usedk_stable and (assign_nmi >= early_cfg.stability_nmi_thr):
            stability_hits += 1
        else:
            stability_hits = 0

        # 6) early stop?
        if (unsup_noimpr_cnt >= early_cfg.unsup_patience_checks) and \
           (stability_hits >= early_cfg.stability_hits_required):
            msg = (
                f"[Early stopping] epoch {epoch}: no SE_spatial/EAS_soft improvement for "
                f"{early_cfg.unsup_patience_checks} checks and assignments stable "
                f"(self-NMI≥{early_cfg.stability_nmi_thr}, UsedK steady over last "
                f"{early_cfg.stability_usedk_window}). "
                f"Best unsup epoch: {best.best_unsup_epoch}."
            )
            if eval_mode and (best.best_nmi is not None):
                msg += f" NMI at best unsup epoch: {best.best_nmi:.6f}."
            print(msg)
            break

    return best