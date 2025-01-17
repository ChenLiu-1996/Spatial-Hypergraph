import cv2
import os
from glob import glob
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
folder_in = '../../data/spatial_placenta_accreta/raw/'
folder_out = '../../data/spatial_placenta_accreta/patchified_data/'
num_bins = 50


if __name__ == '__main__':
    # Find the folders for pixel-by-gene matrices and the corresponding spatial images.
    all_folder_paths = sorted(glob(os.path.join(folder_in, '*', 'filtered_feature_bc_matrix')))
    all_image_paths = sorted(glob(os.path.join(folder_in, '*', 'spatial', 'tissue_hires_image.png')))
    assert len(all_folder_paths) == len(all_image_paths)

    filtered_folder_names, filtered_folders, filtered_image_paths = [], [], []
    for folder_path, image_path in zip(all_folder_paths, all_image_paths):
        if 'normal_placenta' in folder_path or 'PAS' in folder_path or 'insufficient' in folder_path:
            assert ('normal_placenta' in folder_path) + ('PAS' in folder_path) + ('insufficient' in folder_path) == 1
            filtered_folders.append(folder_path)
            filtered_image_paths.append(image_path)
        if 'normal_placenta' in folder_path:
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_normal_placenta')
        elif 'PAS' in folder_path:
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_PAS')
        elif 'insufficient' in folder_path:
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_insufficient')
    del all_folder_paths

    for source_mat_folder, source_image_path, target_folder in tqdm(zip(filtered_folders, filtered_image_paths, filtered_folder_names),
                                                                    total=len(filtered_folders)):
        matrix = ad.io.read_mtx(os.path.join(source_mat_folder, 'matrix.mtx'))
        barcodes = pd.read_csv(os.path.join(source_mat_folder, 'barcodes.tsv'), header=None, sep="\t")
        features = pd.read_csv(os.path.join(source_mat_folder, 'features.tsv'), header=None, sep="\t")
        joined_features = [f"{f0}_{f1}" for f0, f1 in zip(features[0], features[1])]

        barcodes[['X', 'Y']] = barcodes[0].str.extract(r's_\d+um_(\d+)_(\d+)-\d')
        barcodes['X'] = barcodes['X'].astype(int)
        barcodes['Y'] = barcodes['Y'].astype(int)

        # NOTE: Not using the images for now.
        # image = cv2.cvtColor(cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

        # Load the tissue position information.
        tissue_position_info = pd.read_parquet(os.path.join(os.path.dirname(source_image_path), 'tissue_positions.parquet'))
        # Assert the barcode file matches with all pixels that are inside the tissue.
        assert len(barcodes) == tissue_position_info.in_tissue.sum()

        # Subset the data by spatial location.
        cell_bins = pd.DataFrame({'X_bin': pd.cut(barcodes['X'], bins=num_bins, labels=False, include_lowest=True),
                                  'Y_bin': pd.cut(barcodes['Y'], bins=num_bins, labels=False, include_lowest=True),
                                  'X': barcodes['X'],
                                  'Y': barcodes['Y'],
                                  'cell_index': np.arange(len(barcodes))})

        # Iterate over groups and save them separately.
        for (x_bin, y_bin), group in tqdm(cell_bins.groupby(['X_bin', 'Y_bin']), total=num_bins**2):
            # Extract rows corresponding to this group
            indices = group['cell_index'].values
            sub_matrix = matrix.X.T[indices, :]  # Select rows for the group
            sub_adata = ad.AnnData(X=sub_matrix, obs=pd.DataFrame({'Location': group['cell_index']}), var=pd.DataFrame({'Gene Expression': joined_features}))
            coords = np.concatenate((group['X'].values[:, None], group['Y'].values[:, None]), axis=1)
            sub_adata.obsm['spatial'] = coords

            os.makedirs(folder_out, exist_ok=True)
            sub_adata.write(os.path.join(folder_out, f'{target_folder}_Bin-{str(x_bin).zfill(2)}-{str(y_bin).zfill(2)}_spatial_matrix.h5ad'))
