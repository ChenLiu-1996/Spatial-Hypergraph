import anndata as ad
import os
import pathlib

DATA_DIR = '../../data/raw/'
OUTPUT_DIR = '../../data/interim/section_data/'

pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

datasets = os.listdir(DATA_DIR)

for patient in datasets:
    print(DATA_DIR + patient)
    adata_patient = ad.read_h5ad(DATA_DIR + patient)
    sections = adata_patient.obs['Section'].unique()
    for section in sections:
        adata_section = adata_patient[adata_patient.obs['Section'] == section]
        # print(section)
        adata_section.write(f'{OUTPUT_DIR}/{section}.h5ad')
