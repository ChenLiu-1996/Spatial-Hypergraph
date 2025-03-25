import os


data_dir = '../../data/MIBI/raw/'

download_commands = [
    f'wget -O {data_dir}/cell_protein_data.csv https://data.mendeley.com/public-files/datasets/79y7bht7tf/files/2679faaa-53b9-40e1-8315-3e25ede010b7/file_downloaded',
    f'wget -O {data_dir}/cell_spatial_data.csv https://data.mendeley.com/public-files/datasets/79y7bht7tf/files/456465e9-ba45-4846-8c54-df2ffe210cf4/file_downloaded',
    f'wget -O {data_dir}/patient_info.csv https://data.mendeley.com/public-files/datasets/79y7bht7tf/files/ba49ff67-a7a6-45e7-9973-cf975ca5daf2/file_downloaded'
]


if __name__ == '__main__':
    os.makedirs(data_dir, exist_ok=True)
    for command in download_commands:
        os.system(command)