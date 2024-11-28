from glob import glob
from pathlib import Path
import logging
import traceback
import datetime
from typing import List, Dict, Tuple

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
LOGPATH = Path()


def pull_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    metadata['class_vector'] = metadata['classID'].apply(lambda x: one_hot_encode(x,max_id=metadata['classID'].max()))
    return metadata

def initialize_training_objects(model: keras.Model,
                                loss: keras.losses.Loss, 
                                optimizer: keras.optimizers.Optimizer, 
                                training_metrics: List[keras.metrics.Metric], 
                                validation_metrics: List[keras.metrics.Metric]) -> TrainingInstance:
    
    training_instance = TrainingInstance(
                    model = model,
                    optimizer=optimizer,
                    loss_func=loss,
                    training_metrics=training_metrics,
                    validation_metrics=validation_metrics
                 )
    
    return training_instance

def create_tb_log(log_path: Path):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_log_path = log_path / Path(f"{current_time}")
    summary_writer = tf.summary.create_file_writer(training_log_path.__str__())
    return summary_writer

def validation_loop(training_instance: TrainingInstance,
                  metadata: pd.DataFrame,
                  batch_size: int,
                  tb_training_writer: tf.summary.SummaryWriter,
                  epoch: int,
                  fold: int,
                  ):
    fold_val_metadata = metadata.loc[metadata['fold'] == fold, :]
    batched_fold_metadata = random_batch_df(fold_val_metadata, batch_size = batch_size)
    for batch_value in batched_fold_metadata['batch'].unique():
        try:
            batch = batched_fold_metadata[batched_fold_metadata['batch'] == batch_value]
            batch['full_path'] = batch.loc[:, ['slice_file_name', 'fold']].apply(lambda x: URBAN_8K_WAV_PATH / Path(f"fold{x['fold']}") / Path(x['slice_file_name']), axis = 1 )
            batch['x_data'] = batch['full_path'].apply(lambda x: import_audio_from_file(x))
            x_data = tf.stack(batch['x_data'].values.tolist())
            y_data = tf.stack(batch['class_vector'].values.tolist())
            x_data = x_data[:, :, 0]
            y_data = y_data[:, :, 0]
            training_instance.test_step(x_data,  y_data)
        except Exception as e:          
            traceback.print_exc()
            print(f"Error message: {e}")
    validation_metrics = training_instance.get_validation_metric_state()
    for ind, metric in enumerate(validation_metrics): # TODO make this so you can get the name of the metric in the summarizer        
        with tb_training_writer.as_default():
            tf.summary.scalar(f'Val Metric {ind}', metric.numpy(), step=epoch)
    training_instance.reset_validation_metrics()
    return training_instance
    
def training_loop(training_instance: TrainingInstance,
                  metadata: pd.DataFrame,
                  epochs: int, 
                  batch_size: int,
                  tb_training_writer: tf.summary.SummaryWriter,
                  fold: int = 1,
                  tb_logging_interval: int = 10, 
                  ) -> TrainingInstance:
    step = 0
    for epoch in range(0,epochs):
        fold_train_metadata = metadata.loc[metadata['fold'] != fold, :]
        batched_fold_metadata = random_batch_df(fold_train_metadata, batch_size= batch_size)
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
                if step % tb_logging_interval == 0:
                    with tb_training_writer.as_default():
                        tf.summary.scalar('training loss per batch', loss_val.numpy(), step=step)
                step += 1
            except Exception as e:          
                traceback.print_exc()
                print(f"Error message: {e}")        
        training_metrics = training_instance.get_train_metric_state()
        for ind, metric in enumerate(training_metrics): # TODO make this so you can get the name of the metric in the summarizer        
            with tb_training_writer.as_default():
                tf.summary.scalar(f'Training Metric {ind}', metric.numpy(), step=epoch)
        training_instance.reset_training_metrics()
        training_instance = validation_loop(training_instance, 
                                            metadata=metadata, 
                                            batch_size=batch_size, 
                                            tb_training_writer=tb_training_writer, 
                                            epoch = epoch, 
                                            fold = fold)
        print('completed epoch')
    
    return training_instance



# Pull metadata
metadata = pull_metadata(URBAN_8K_METADATA_PATH)

# Make Summary Writer
tb_summary_writer = create_tb_log(log_path=r'/mnt/d/2024/proj/BIDRCLEF_2024/logs')

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

training_loop(training_instance=training_instance, 
              metadata=metadata,
              epochs = EPOCHS,
              tb_training_writer=tb_summary_writer, 
              batch_size=BATCH_SIZE)
