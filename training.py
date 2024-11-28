from glob import glob
from pathlib import Path
import logging
import traceback
import datetime

import tensorflow as tf
from tensorflow import keras
import pandas as pd

from model_training.data_import.data_import_utils import one_hot_encode, random_batch_df, import_audio_from_file
from model_training.models.model import TempFreqAudioClassificationModel
from model_training.trainer.trainer import TrainingInstance

#TODO Add necessary functions to wrap training in an intelligent way.

# Global variables
EPOCHS = 100
BATCH_SIZE = 16
URBAN_8K_WAV_PATH = Path(r'/mnt/d/2024/proj/BIDRCLEF_2024/UrbanSound8K/audio')
URBAN_8K_METADATA_PATH = Path(r'/mnt/d/2024/proj/BIDRCLEF_2024/UrbanSound8K/metadata/UrbanSound8K.csv')

# Pull metadata
metadata = pd.read_csv(URBAN_8K_METADATA_PATH)
# Encode the class ID
metadata['class_vector'] = metadata['classID'].apply(lambda x: one_hot_encode(x,max_id=metadata['classID'].max()))
# Get fold count for cross validation procedure
cross_validation_folds = metadata['fold'].max()

# create model, training instance
model = TempFreqAudioClassificationModel(num_classes=10)
loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.Adam()
training_metrics = [keras.metrics.CategoricalAccuracy()]
validation_metrics = [keras.metrics.CategoricalAccuracy()] 
training_instance = TrainingInstance(
                    model = model,
                    optimizer=optimizer,
                    loss_func=loss,
                    training_metrics=training_metrics,
                    validation_metrics=validation_metrics
                 )

# Create tensorboard log
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
training_log_path = Path('/mnt/d/2024/proj/BIDRCLEF_2024/logs') / Path(f"{current_time}")
training_summary_writer = tf.summary.create_file_writer(training_log_path.__str__())

# Run training across each fold -> turn this off if you want to. Maybe turn this into a function.
# for fold in range(0, cross_validation_folds):
fold = 0
losses = []
train_acc = []
val_acc = []
step = 0
for epoch in range(0,EPOCHS):
    fold_train_metadata = metadata.loc[metadata['fold'] != fold, :]
    fold_test_metadata = metadata.loc[metadata['fold'] == fold, :]
    fold_test_metadata['full_path'] = fold_test_metadata[['slice_file_name', 'fold']].apply(lambda x: URBAN_8K_WAV_PATH / Path(f"fold{x['fold']}") / Path(x['slice_file_name']), axis = 1 )
    fold_test_metadata['x_data'] = fold_test_metadata['full_path'].apply(lambda x: import_audio_from_file(x))
    x_data_test = tf.stack(fold_test_metadata['x_data'].values.tolist())
    y_data_test = tf.stack(fold_test_metadata['class_vector'].values.tolist())
    x_data_test = x_data_test[:, :, 0]
    y_data_test = y_data_test[:, :, 0]
    print('hey!')
    print(y_data_test.shape)
    batched_fold_metadata = random_batch_df(fold_train_metadata, batch_size= BATCH_SIZE)
    for batch_value in batched_fold_metadata['batch'].unique():
        try:
            batch = batched_fold_metadata[batched_fold_metadata['batch'] == batch_value]
            batch['full_path'] = batch.loc[:, ['slice_file_name', 'fold']].apply(lambda x: URBAN_8K_WAV_PATH / Path(f"fold{x['fold']}") / Path(x['slice_file_name']), axis = 1 )
            batch['x_data'] = batch['full_path'].apply(lambda x: import_audio_from_file(x))
            x_data = tf.stack(batch['x_data'].values.tolist())
            y_data = tf.stack(batch['class_vector'].values.tolist())
            x_data = x_data[:, :, 0]
            y_data = y_data[:, :, 0]
            loss_val = training_instance.train_step(x_data,  y_data)
            losses.append(loss_val.numpy())
            if step % 100 == 0:
                with training_summary_writer.as_default():
                    tf.summary.scalar('training loss', loss_val.numpy(), step=step)
            step += 1
        except Exception as e:          
            # Print the full traceback
            traceback.print_exc()
            # Optionally, log or work with the exception details
            print(f"Error message: {e}")        
    with training_summary_writer.as_default():
        training_acc = training_instance.get_train_metric_state()
        training_instance.reset_training_metrics()
        training_instance.test_step(x_data,  y_data)
        training_instance.test_step()
        tf.summary.scalar('training accuracy', training_acc[0].numpy(), step=epoch)
        tf.summary.scalar('validation accuracy', training_acc[0].numpy(), step=epoch)
        print('completed epoch')
            

