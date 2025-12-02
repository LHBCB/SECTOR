from __future__ import annotations
from copy import deepcopy
from typing import Dict, Any, Iterable, Optional

from . import fit as fit_mod
from . import pred as pred_mod
from .utility.parser import parse_args as _parse_args

from argparse import Namespace

class SECTOR:
    """
    Orchestrator that reuses existing fit.py / pred.py logic.
    Supports per-call overrides: m.fit(epochs=600, lr=5e-4, persist=True), etc.

    Nothing in here changes the model math—this only wires state together.
    """

    # Keys that imply data/model/device must be rebuilt if changed
    DATASET_KEYS: set[str] = {
        "dataset_path", "dataset", "slice", "label", "weight_mode", "n_comps", "n_top_genes"
    }
    DEVICE_KEYS: set[str] = {"gpu"}
    MODEL_KEYS: set[str] = {
        "embed_dim", "num_clusters", "dropout", "activation", "k", "beta_f", "seed"
    }
    # Prediction-only knobs that never require rebuild
    PRED_ONLY_KEYS: set[str] = {
        "island_min_abs", "island_min_frac", "island_max_iter", "root_cluster", "spatial_anchor"
    }
    # Optimizer-only “hot” override
    OPT_ONLY_KEYS: set[str] = {"lr"}

    def __init__(self, args: Optional[Any] = None, **init_overrides):
        """
        If args is None, pull defaults via parse_args([]).
        You can also pass a dict for args, or extra keyword overrides.
        """
        # Build a base Namespace
        if args is None:
            base = _parse_args([])                 # defaults only
        elif isinstance(args, dict):
            base = _parse_args([])                 # start from defaults, then apply dict
            for k, v in args.items():
                setattr(base, k, v)
        elif isinstance(args, Namespace):
            base = deepcopy(args)
        else:
            # any object with attributes works
            base = deepcopy(args)
    
        for k, v in init_overrides.items():
            setattr(base, k, v)
    
        fit_mod.set_seed(base)
        self.args = base
        self.device, self.dataset, self.model_path = fit_mod.setup_device_and_data(self.args)
        self.lambda_tv_target, self.start_epoch, self.end_epoch = fit_mod.setup_tv_warmup(self.args)
        self.model, self.optimizer = fit_mod.init_model_and_optimizer(self.args, self.dataset, self.device)
        self.fit_result: Optional[fit_mod.FitResult] = None
    
        # -------------------------
        # Internal helpers
        # -------------------------
    
    @staticmethod
    def _clone_args(args):
        return deepcopy(args)

    @staticmethod
    def _apply_overrides(args, overrides: Dict[str, Any]):
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def _needs_rebuild(self, overrides: Dict[str, Any], for_pred: bool = False) -> bool:
        keys = set(overrides.keys())
        # Anything that touches dataset/device/model requires rebuild
        if keys & (self.DATASET_KEYS | self.DEVICE_KEYS | self.MODEL_KEYS):
            return True
        # For pred, most overrides do not require rebuild
        return False

    def _rebuild_everything(self, args):
        # Recreate dataset/model/optimizer exactly like in __init__
        fit_mod.set_seed(args)
        self.device, self.dataset, self.model_path = fit_mod.setup_device_and_data(args)
        self.lambda_tv_target, self.start_epoch, self.end_epoch = fit_mod.setup_tv_warmup(args)
        self.model, self.optimizer = fit_mod.init_model_and_optimizer(args, self.dataset, self.device)

    def _maybe_update_optimizer_hot_params(self, args, overrides: Dict[str, Any]):
        if "lr" in overrides and hasattr(self, "optimizer") and self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = float(args.lr)

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, persist: bool = False, **overrides) -> fit_mod.FitResult:
        """
        Train with optional per-call overrides.

        Examples:
            m.fit(epochs=800, lr=5e-4)
            m.fit(lambda_tv=0.2, tv_warmup_epochs=50, persist=True)
            m.fit(dataset="MouseBrain", slice="S1", seed=123, persist=True)  # triggers rebuild
        """
        # Compose call-local args
        args_fit = self._clone_args(self.args)
        if overrides:
            self._apply_overrides(args_fit, overrides)

        # Rebuild if overrides affect data/device/model/seed
        if self._needs_rebuild(overrides, for_pred=False):
            self._rebuild_everything(args_fit)
        else:
            # Hot params can be changed without rebuild
            self.lambda_tv_target, self.start_epoch, self.end_epoch = fit_mod.setup_tv_warmup(args_fit)
            self._maybe_update_optimizer_hot_params(args_fit, overrides)

        # Optional balance probe (identical logic)
        self.model, self.optimizer, restart_epoch = fit_mod.maybe_run_balance_probe(
            args=args_fit,
            model=self.model,
            optimizer=self.optimizer,
            dataset=self.dataset,
            device=self.device,
            lambda_tv_target=self.lambda_tv_target,
            start_epoch=self.start_epoch,
            end_epoch=self.end_epoch,
        )

        # Early-stopping config + training loop (identical)
        early_cfg = fit_mod.cfg_from_args(args_fit)
        best_state = fit_mod.run_training_loop(
            args=args_fit,
            model=self.model,
            optimizer=self.optimizer,
            dataset=self.dataset,
            lambda_tv_target=self.lambda_tv_target,
            start_epoch=self.start_epoch,
            end_epoch=self.end_epoch,
            restart_epoch=restart_epoch,
            model_path=self.model_path,
            device=self.device,
            early_cfg=early_cfg,
        )

        # Update fit_result
        self.fit_result = fit_mod.FitResult(
            model=self.model,
            dataset=self.dataset,
            device=self.device,
            model_path=self.model_path,
            best_state=best_state,
        )

        # Optionally persist overrides onto the live args
        if persist and overrides:
            self.args = args_fit

        #return self.fit_result

    def pred(self, persist: bool = False, **overrides):
        """
        Predict/export with optional per-call overrides.

        Examples:
            m.pred(island_min_abs=50, island_min_frac=0.02)
            m.pred(root_cluster=3)               # orient pseudotime by a predicted cluster
            m.pred(spatial_anchor="north")       # orient pseudotime by spatial anchor
            m.pred(dataset="MouseBrain", slice="S2", persist=True)  # rebuild on new data
        """
        # Compose call-local args
        args_pred = self._clone_args(self.args)
        if overrides:
            self._apply_overrides(args_pred, overrides)

        # Rebuild if the user touched dataset/device/model knobs
        if self._needs_rebuild(overrides, for_pred=True):
            self._rebuild_everything(args_pred)
            # If pred before fit, create a minimal FitResult
            if self.fit_result is None:
                self.fit_result = fit_mod.FitResult(
                    model=self.model,
                    dataset=self.dataset,
                    device=self.device,
                    model_path=self.model_path,
                    best_state=fit_mod.BestState(),
                )
            else:
                # Keep best_state but ensure it points to current model/dataset
                self.fit_result = fit_mod.FitResult(
                    model=self.model,
                    dataset=self.dataset,
                    device=self.device,
                    model_path=self.model_path,
                    best_state=self.fit_result.best_state or fit_mod.BestState(),
                )
        else:
            if self.fit_result is None:
                # Build a minimal FitResult referencing current state
                self.fit_result = fit_mod.FitResult(
                    model=self.model,
                    dataset=self.dataset,
                    device=self.device,
                    model_path=self.model_path,
                    best_state=fit_mod.BestState(),
                )

        result = pred_mod.infer_clusters_and_pseudotime(args_pred, self.fit_result)

        # Optionally persist overrides
        if persist and overrides:
            self.args = args_pred

        return result

    @property
    def adata(self):
        return self.dataset.adata
