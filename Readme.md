


## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name scdata pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate scdata
conda install scikit-learn scikit-image pandas matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge
python -m pip install phate
python -m pip install dhg anndata torch_geometric einops pytorch_lightning geovoronoi vendi-score

export PROJ_DIR=/usr/local
export PROJ_LIBDIR=/usr/local/lib/
export PROJ_INCDIR=/usr/local/includes/
brew install proj
conda install -c conda-forge pyproj

python -m pip install squidpy
python -m pip install opencv-python


```

## Debug
If you encounter `undefined symbol: cublasLtHSHMatmulAlgoInit, version libcublasLt.so.11`, can try the following.
```
export LD_LIBRARY_PATH=/home/cl2482/.conda/envs/scdata/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
```
NOTE: replace `/home/cl2482/.conda/envs/scdata/lib/python3.10/` with your environment parent directory.
