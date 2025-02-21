import logging
import os
import anndata as ad
import scanpy as sc
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.neighbors import kneighbors_graph

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class PlacentaDataset(Dataset):
    '''
    Placenta Dataset.
    Spatial RNA-seq data on placenta.
    Data are given in matrices of matrices of [pixel coordinates, gene expression].
    We have chopped up the data into small neighborhoods.
    The purpose is to classify these neighborhoods into 3 classes:
        - Normal
        - Placenta Accreta Spectrum (PAS)
        - Placental Insufficiency

    Returned `graph_data`: a torch_geometric.data.Data instance, where
                           graph_data.x is the node features.
    '''

    def __init__(self,
                 data_folder: str = '../../data/spatial_placenta_accreta/patchified/',
                 transform=None):

        self._load_data(data_folder)
        self.transform = transform

    def _load_data(self, data_folder: str) -> None:
        graph_path_list = sorted(glob(data_folder + '*.h5ad'))
        class_list = []
        self.class_map = {
            0: 'normal',
            1: 'PAS',
            2: 'insufficient',
        }

        for graph_path in graph_path_list:
            graph_str = os.path.basename(graph_path)
            if 'normal' in graph_str:
                class_list.append(0)
            elif 'PAS' in graph_str:
                class_list.append(1)
            elif 'insufficient' in graph_str:
                class_list.append(2)
            else:
                raise ValueError(f'graph_str must contain `normal`, `PAS` or `insufficient`, but got {graph_str}.')

        assert len(graph_path_list) == len(class_list)

        self.graph_path_arr = np.array(graph_path_list)
        self.class_arr = np.array(class_list)
        return

    def __len__(self) -> int:
        return len(self.graph_path_arr)

    def __getitem__(self, idx: int) -> Data:
        adata = ad.read_h5ad(self.graph_path_arr[idx])
        graph_data = return_graph_data(adata)
        y_true = self.class_arr[idx]
        graph_data.y = y_true

        if self.transform:
            graph_data = self.transform(graph_data)
        return graph_data

def return_graph_data(adata):
    # Normalize the gene expression for each pixel.
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    # Create the graph.
    G = create_knn_graph(adata)

    # NetworkX to PyG.
    data = from_networkx(G)
    data.x = torch.tensor(adata.X.todense(), dtype=torch.float)
    return data

def create_knn_graph(adata, K: int = 5):
    sparseA = kneighbors_graph(adata.obsm['spatial'], n_neighbors=K, mode='connectivity', include_self=False)
    A = sparseA.todense()
    G = nx.from_numpy_array(A)
    return G


if __name__ == '__main__':
    dataset = PlacentaDataset()
