import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists
from tqdm import tqdm


DATA_DIR = 'E:\Data\ADL_Dataset\HMP_Dataset'.replace('\\', '/')
OUT_FOLDER = '../output'
DATA_FREQUENCY = '31.25ms'
EPOCH_1S = 32
EPOCH_5S = EPOCH_1S * 5


def transform_dataframe(input_df, activity, vol_id):

    aggr_df = pd.DataFrame()
    aggr_df['vm_mean'] = input_df['vm'].mean()
    aggr_df['menmo'] = input_df['enmo'].mean()
    aggr_df['vm_sd'] = input_df['vm'].std()
    aggr_df['vm_max'] = input_df['vm'].max()
    aggr_df['vm_min'] = input_df['vm'].min()
    aggr_df['vm_10perc'] = input_df['vm'].quantile(.1)
    aggr_df['vm_25perc'] = input_df['vm'].quantile(.25)
    aggr_df['vm_50perc'] = input_df['vm'].quantile(.5)
    aggr_df['vm_75perc'] = input_df['vm'].quantile(.75)
    aggr_df['vm_90perc'] = input_df['vm'].quantile(.9)
    aggr_df['activity'] = activity
    aggr_df['vol_id'] = vol_id

    return aggr_df


def revert_to_acceleration(row, attr):
    return -1.5 * 3*(row[attr]/63)


if __name__ == '__main__':

    # Load all the folders
    folders = [f for f in listdir(DATA_DIR) if isdir(join(DATA_DIR, f))]

    # Result dataframe
    col_names = ['vm_mean', 'menmo', 'vm_sd', 'vm_max', 'vm_min', 'vm_10perc',
                 'vm_25perc', 'vm_50perc', 'vm_75perc', 'vm_90perc', 'activity', 'vol_id']
    result_df_1s = pd.DataFrame(columns=col_names)
    result_df_5s = pd.DataFrame(columns=col_names)

    # Loop through all the folders
    for folder_name in folders:

        folder_path = join(DATA_DIR, folder_name)

        # Extract files inside the folder
        files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

        # Loop through the files
        for file_name in tqdm(files, desc=('Processing '+folder_name)):

            file_path = join(folder_path, file_name)
            file_split = file_name.split('-')

            # Read file
            file_df = pd.read_csv(file_path, header=None, delimiter=' ', names=['x', 'y', 'z'])

            # Compose secondary attributes
            file_df['x'] = file_df.apply(revert_to_acceleration, attr='x', axis=1)
            file_df['y'] = file_df.apply(revert_to_acceleration, attr='y', axis=1)
            file_df['z'] = file_df.apply(revert_to_acceleration, attr='z', axis=1)
            file_df['vm'] = np.sqrt(file_df['x'] ** 2 + file_df['y'] ** 2 + file_df['z'] ** 2)
            file_df['enmo'] = file_df['vm'] - 1
            file_df.loc[file_df['enmo'] < 0, 'enmo'] = 0

            # Average to get 1 sec epoch
            file_epoch_1s_df = file_df.groupby(np.arange(len(file_df)) // EPOCH_1S)
            aggr_1s_df = transform_dataframe(file_epoch_1s_df, file_split[-2], file_split[-1].replace('.txt', ''))
            result_df_1s = result_df_1s.append(aggr_1s_df, sort=False)

            # Average to get 5 sec epoch
            file_epoch_5s_df = file_df.groupby(np.arange(len(file_df)) // EPOCH_5S)
            aggr_5s_df = transform_dataframe(file_epoch_5s_df, file_split[-2], file_split[-1].replace('.txt', ''))
            result_df_5s = result_df_5s.append(aggr_5s_df, sort=False)

        # End file loop

    # End folder loop

    # Output folder
    if not exists(OUT_FOLDER):
        makedirs(OUT_FOLDER)

    # Save the output
    result_df_1s.to_csv(join(OUT_FOLDER, 'adl_aggr_1s.csv'), index=None)
    result_df_5s.to_csv(join(OUT_FOLDER, 'adl_aggr_5s.csv'), index=None)

    print('Completed.')
