import argparse

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
    parser.add_argument('--k_s', type=int, default=6,
                        help='Number of neighbours used to create spatial graph.')
    parser.add_argument('--k', type=int, default=1,
                        help='Number of neighbours used to create feature graph.')

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

    # ================================
    # Early stopping (label-free)
    # ================================
    parser.add_argument("--unsup_patience_checks", type=int, default=6,
        help="How many consecutive 'verbose checks' with no improvement in SE_spatial/EAS_soft to tolerate before stopping (also requires stability hits).")
    parser.add_argument("--rel_improve_tol", type=float, default=0.005,
        help="Relative improvement tolerance for SE_spatial (lower is better) and EAS_soft (higher is better).")
    parser.add_argument("--stability_nmi_thr", type=float, default=0.97, help="NMI threshold between consecutive hard assignments to count as 'stable'.")
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