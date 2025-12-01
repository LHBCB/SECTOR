#!/usr/bin/env python3
"""
SECTOR command-line runner

Runs training + clustering/trajectory + plotting (optional) in one go.

USAGE
-----
python run_sector.py --dataset_path ./data --dataset DLPFC --slice 151673 --num_clusters 7 --plot 1
"""

import sys

from sector import SECTOR
from sector.utility.parser import parse_args as sector_parse_args


def _print_help_and_exit() -> None:
    print(__doc__)
    sys.exit(0)


def main():
    argv = sys.argv[1:]
    if not argv:
        flags = []
    else:
        first = argv[0].lower()
        if first in ("-h", "--help"):
            _print_help_and_exit()
        else:
            flags = argv

    # Parse SECTOR flags and run
    args = sector_parse_args(flags)
    m = SECTOR(args)

    print("[SECTOR] Fitting...")
    m.fit(persist=True)

    print("[SECTOR] Predicting / exporting...")
    m.pred(persist=True)

    # Convenience: echo expected outputs
    ckpt = f"./sector_model/{args.dataset}_{args.slice}_K{args.num_clusters}.pt"
    h5ad = f"./output/{args.dataset}.{args.slice}.sector.h5ad"
    fig_clusters = f"./figures/{args.dataset}.{args.slice}.clusters.png"
    fig_ptime = f"./figures/{args.dataset}.{args.slice}.pseudotime.png"
    
    print("\n[SECTOR] Finished clustering and pseudotime ordering.")
    print("  Checkpoint: ", ckpt)
    
    if args.save_adata:
        print("  H5AD:       ", h5ad)
    if args.plot:
        print("  Figures:    ", fig_clusters, " , ", fig_ptime)


if __name__ == "__main__":
    main()