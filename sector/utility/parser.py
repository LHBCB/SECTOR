import argparse

def _str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        value = v.strip().lower()
        if value in {"true", "t", "1", "yes", "y"}:
            return True
        if value in {"false", "f", "0", "no", "n"}:
            return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run.")

    # ================================
    # ST data
    # ================================
    parser.add_argument('--dataset_path', type=str, default='./data', 
                        help='Path of spatial transcriptomics datasets.')
    parser.add_argument('--dataset', nargs='?', default='DLPFC', 
                        help='Choose a dataset from {DLPFC, MERFISH, STARmap, BaristaSeq, StereoSeq, or else}.')
    parser.add_argument('--slice', nargs='?', default='151673', 
                        help='Choose a slice from the dataset.')
    parser.add_argument('--label', type=str, default='Region',
                        help='Ground truth column in adata.obs (required for eval_mode=1).') 
    parser.add_argument('--n_comps', type=int, default=20,
                       help='Number of PCs')
    parser.add_argument('--n_top_genes', type=int, default=2000,
                       help='Number of HVGs.')
    parser.add_argument('--use_svg', type=_str2bool, nargs='?', const=True, default=False,
                       help='Use SVGs instead of HVGs for feature selection (True/False).')
    parser.add_argument('--k_s', type=int, default=6,
                        help='Number of neighbours used to create spatial graph.')
    parser.add_argument('--k', type=int, default=1,
                        help='Number of neighbours used to create feature graph.')

    parser.add_argument('--large_scale_mode', type=str, default='auto',
                        help="Switching rule for large datasets: {'auto', 'off', 'on'}. In auto mode: n_obs < 10000 -> dense mode; 10000 <= n_obs < large_scale_n_obs_threshold -> sparse mode with attr_graph_source='mlp'; n_obs >= large_scale_n_obs_threshold -> sparse mode with attr_graph_source='raw'. Manual 'off' / 'on' preserve the dense / sparse override behavior.")
    parser.add_argument('--large_scale_n_obs_threshold', type=int, default=100000,
                        help='Upper cutoff used by auto mode: above this, sparse mode keeps effective attr_graph_source=\'raw\'; between 10000 and this threshold, sparse mode keeps effective attr_graph_source=\'mlp\'.')
    parser.add_argument('--use_hvg_only', type=int, default=1,
                        help='Only used in large-scale mode: 1 keeps HVGs only, 0 keeps all genes.')
    parser.add_argument('--scale_before_pca', type=int, default=0,
                        help='Only used in large-scale mode: whether to scale before PCA.')
    parser.add_argument('--pca_zero_center', type=int, default=0,
                        help='Only used in large-scale mode: whether PCA should zero-center dense matrices.')
    parser.add_argument('--attr_graph_mode', type=str, default='cached_exact',
                        help='Only used in large-scale mode: feature graph builder {cached_exact, dense_exact, off}.')
    parser.add_argument('--attr_graph_source', type=str, default='mlp',
                        help="Feature graph source {mlp, raw}. In dense mode, mlp reproduces the original mlp behavior and raw switches the dense feature graph to use raw PCA features. In auto sparse mode, the effective source is forced by dataset size (<threshold: mlp, >=threshold: raw). Manual large_scale_mode='on' keeps this user setting.")
    parser.add_argument('--attr_knn_algorithm', type=str, default='auto',
                        help='Only used in large-scale mode: sklearn NearestNeighbors algorithm {auto, kd_tree, ball_tree, brute}.')
    parser.add_argument('--attr_leaf_size', type=int, default=40,
                        help='Only used in large-scale mode: leaf size for sklearn NearestNeighbors.')
    parser.add_argument('--attr_n_jobs', type=int, default=-1,
                        help='Only used in large-scale mode: CPU workers for cached feature graph build.')
    parser.add_argument('--attr_dense_exact_max_nodes', type=int, default=50000,
                        help='Only used in large-scale mode: safety cap if attr_graph_mode=dense_exact.')

    # ================================
    # Model fitting
    # ================================
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index; use cuda:0 by default.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--num_clusters', type=int, default=7,
                        help='Number of expected clusters.')
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--activation', type=str, default='relu',
                        help='elu, relu, sigmoid, None.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate.')
    parser.add_argument('--beta_f', type=float, default=0.5,
                        help='Weight for feature graph in fused graph.')
    parser.add_argument('--patience', type=int, default=300,
                        help='Stop training if no improvement for patience epochs.')
    parser.add_argument('--weight_mode', type=str, default='gaussian',
                        help='Build weighted spatial graph, can be {gaussian, inverse, binary}.')
    parser.add_argument('--eval_mode', type=int, default=1,
                        help='1: evaluate clustering performance (adata should have labels); 0: no evaluation.')
    parser.add_argument('--lambda_tv', type=float, default=2.0,
                        help='TV regularizer coefficient.')
    parser.add_argument('--tv_warmup_epochs', type=int, default=100,
                        help='TV warmup epochs.')
    parser.add_argument('--gamma_balance', type=float, default=1.0,
                        help='Balance regularizer coefficient (optional).')
    parser.add_argument('--balance_mode', type=str, default="volume",
                        help='Volume or node mode for balancing.')
    parser.add_argument('--balance_probe_epochs', type=int, default=20,
                        help='Balance probe epochs: restart training if not matching expected number of clusters.')
    parser.add_argument('--verbose', type=int, default=20,
                        help='For probe running, evaluate every verbose epochs.')

    parser.add_argument('--detect_anomaly', type=int, default=0,
                        help='Only used in large-scale mode: whether to keep autograd anomaly detection on.')

    # ================================
    # Early stopping (label-free)
    # ================================
    parser.add_argument("--unsup_patience_checks", type=int, default=6,
        help="How many consecutive 'verbose checks' with no improvement in SE_spatial/EAS_soft to tolerate before stopping (also requires stability hits).")
    parser.add_argument("--rel_improve_tol", type=float, default=0.005,
        help="Relative improvement tolerance for SE_spatial (lower is better) and EAS_soft (higher is better).")
    parser.add_argument("--stability_nmi_thr", type=float, default=None,
        help="NMI threshold between consecutive hard assignments to count as 'stable'. If omitted, it is set automatically by cell count: <10000 -> 0.97, 10000-100000 -> 0.999, >100000 -> 1.0.")
    parser.add_argument("--stability_usedk_window", type=int, default=4,
        help="Window size (in verbose checks) over which the number of used clusters (UsedK) must remain constant.")
    parser.add_argument("--stability_hits_required", type=int, default=3,
        help="Consecutive stable hits required (NMI≥thr and UsedK steady) to allow early stopping after patience is exceeded.")

    # ================================
    # Post hoc island cleaner
    # ================================
    parser.add_argument("--island_min_frac", type=float, default=0.0,
        help="Minimum component size as a fraction of the largest component of that label.")
    parser.add_argument("--island_min_abs", type=int, default=0,
        help="Absolute minimum component size to keep per cluster label.")
    parser.add_argument("--island_max_iter", type=int, default=2,
        help="Maximum number of cleanup passes.")

    # ================================
    # Pseudotime
    # ================================
    parser.add_argument('--root_cluster', type=int, default=None,
        help='Root cluster for pseudotime orientation.')
    parser.add_argument('--spatial_anchor', type=str, default='south',
        help='One of north, south, east, west.')
    
    # ================================
    # Plot, save, and others
    # ================================
    parser.add_argument('--plot', type=bool, default=False,
        help='Plot and save spatial clusters and pseudotime or not.')
    parser.add_argument('--invert_y', type=bool, default=True,
        help='Invert y axis or not, based on spatial coordinate conventions.')
    parser.add_argument('--save', type=bool, default=True,
        help='Save trained sector model or not.')
    parser.add_argument('--save_adata', type=bool, default=True,
        help='Save adata or not.')

    if argv is None:
        # CLI / notebook path: read sys.argv and safely ignore stray Jupyter flags
        return parser.parse_known_args()[0]
    else:
        # Programmatic path: parse only what pass in
        return parser.parse_known_args(argv)[0]