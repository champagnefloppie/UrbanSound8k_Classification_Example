from glob import glob
from pathlib import Path
import logging


import tensorflow as tf
import pandas as pd
from model_training.data_import.data_import_utils import one_hot_encode, random_batch_df, import_audio_from_file

# Global variables
EPOCHS = 100
URBAN_8K_WAV_PATH = Path(r'/mnt/d/2024/proj/BIDRCLEF_2024/UrbanSound8K/audio')

# Pull metadata
metadata = pd.read_csv('/mnt/d/2024/proj/BIDRCLEF_2024/UrbanSound8K/metadata/UrbanSound8K.csv')
# Encode the class ID
metadata['class_vector'] = metadata['classID'].apply(lambda x: one_hot_encode(x,max_id=metadata['classID'].max()))
# Get fold count for cross validation procedure
cross_validation_folds = metadata['fold'].max()


# Run training across each fold -> turn this off if you want to. Maybe turn this into a function.
for fold in range(0, cross_validation_folds):
    for epoch in range(0,EPOCHS):
        fold_train_metadata = metadata.loc[metadata['fold'] != fold, :]
        fold_test_metadata = metadata.loc[metadata['fold'] == fold, :]
        batched_fold_metadata = random_batch_df(fold_train_metadata, batch_size= 128)
        for batch_value in batched_fold_metadata['batch'].unique():
            batch = batched_fold_metadata[batched_fold_metadata['batch'] == batch_value]
            batch['full_path'] = batch.loc[:, ['slice_file_name', 'fold']].apply(lambda x: URBAN_8K_WAV_PATH /Path(f"fold{x['fold']}")/Path(x['slice_file_name']), axis = 1 )
            batch['x_data'] = batch['full_path'].apply(lambda x: import_audio_from_file(x))
            
            print(tf.stack(batch['x_data'].values.tolist()))
            break
        break
    break
