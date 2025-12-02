# SECTOR

---

## 1. Clone the repo
```bash
git clone https://github.com/LHBCB/SECTOR.git
cd SECTOR
```

## 2. Installation
SECTOR has been developed and evaluated with `Python 3.12`, `PyTorch 2.7.1` with `CUDA 12.6`, and `torch_geometric 2.7.0`. We recommend using a dedicated conda environment.
### 2.1. Create a conda environment

```bash
conda create -n sector_env python=3.12
conda activate sector_env
```

### 2.2. Install PyTorch (CUDA build; recommended)
```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```
If you prefer a CPU-only installation or a different CUDA version, please follow the official PyTorch instructions (https://pytorch.org) for your system. Once PyTorch is installed, return here for the remaining dependencies.

### 2.3. Install SECTOR dependencies
From the repository root:
```bash
pip install -r requirements.txt
```
This installs, among others: 
- Core scientific libraries: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `scikit-misc`
- ST / AnnData ecosystem: `anndata`, `scanpy`, `h5py`
- GNN stack (on top of installed PyTorch): `torch-geometric`
- Notebook support: `notebook`, `ipykernel`

## 3. Running SECTOR from the command line
SECTOR expects data in `.h5ad` (AnnData) format. By default, both the CLI and Python API look for files at:
```bash
{dataset_path}/{dataset}/{slice}.h5ad
```
For example, the 10x Visium DLPFC slice used in the tutorial (see our paper for detailed information about this dataset) is expected at:
```bash
./data/DLPFC/151673.h5ad
```
You can change `--dataset_path`, `--dataset` and `--slice` (or the corresponding arguments in the Python API) to point to your own datasets.

The main CLI entry point is `run_sector.py`. A typical run on a DLPFC slice is:
```bash
python  run_sector.py \
    --dataset_path ./data \
    --dataset DLPFC \
    --slice 151673 \
    --num_clusters 7 \
    --lambda_tv 2.0 \
    --eval_mode 1 \
    --plot True \
    --island_min_frac 0.1 \
    --island_min_abs 40
```

## 4. Running SECTOR from the Python API
SECTOR can also be used directly from Python, for example in Jupyter notebooks. See the tutorial notebook [tutorial_DLPFC.ipynb](tutorial_DLPFC.ipynb), which demonstrates:
- How to initialise and train a SECTOR model using the Python API.
- How to infer and visualise the spatial domains and pseudotime.
- How to inspect evaluation metrics (when input data contain labels).
