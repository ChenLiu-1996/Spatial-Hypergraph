

## Reproducing the results
1. Preprocess the data

1.1 Yang-Hartwich data.

1.1.1 If using all the genes.
```
cd src/preprocessing
python preprocess_placenta_patchify_all_genes.py
```

1.1.2 If using a selected subset of genes of interest.
```
cd src/preprocessing
python preprocess_placenta_patchify_selected_genes.py
```

1.2 MIBI data.

1.2.1 If using all genes.
```
cd src/preprocessing
python preprocess_MIBI_patchify_all_genes.py
```

2. Train the hypergraph wavelet model.
2.1 If using all the genes.
```
cd src
python train.py --data-folder '$ROOT/data/MIBI/patchified_all_genes/'
```

2.2 If using a selected subset of genes of interest.
```
cd src
python train.py --data-folder '$ROOT/data/MIBI/patchified_selected_genes/'
```


## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name scdata pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c anaconda -c conda-forge -y
conda activate scdata
conda install scikit-learn scikit-image pandas matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge -y
python -m pip install opencv-python
python -m pip install phate
python -m pip install meld
python -m pip install anndata torch_geometric einops geovoronoi vendi-score
python -m pip install numpy==1.26
python -m pip install pyarrow
python -m pip install transformers
python -m pip install dhg==0.9.4 --no-deps
python -m pip install optuna torchmetrics scanpy
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
python -m pip install git+https://huggingface.co/ctheodoris/Geneformer

export PROJ_DIR=/usr/local
export PROJ_LIBDIR=/usr/local/lib/
export PROJ_INCDIR=/usr/local/includes/
brew install proj
conda install -c conda-forge pyproj
python -m pip install squidpy


```

## Debug
If you encounter `undefined symbol: cublasLtHSHMatmulAlgoInit, version libcublasLt.so.11`, can try the following.
```
export LD_LIBRARY_PATH=/home/cl2482/.conda/envs/scdata/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
```
NOTE: replace `/home/cl2482/.conda/envs/scdata/lib/python3.10/` with your environment parent directory.
