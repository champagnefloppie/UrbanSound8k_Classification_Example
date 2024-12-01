from glob import glob
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio


def zero_pad_tensor(audio_tensor: tf.Tensor, length: int = 88200) -> tf.Tensor:
    """Function to zero pad an audio tensor along its audio channel (first dimension)

    Args:
        audio_tensor (tf.Tensor): Input audio tensor of shape (input_length, channels).
        length (int, optional): Desired length to pad end of audio to. Defaults to 88200.

    Returns:
        tf.Tensor: Padded audio tensor.
    """
    padding = [[0, length - audio_tensor.shape[0]], [0, 0]]  # No padding for the second dimension
    padded_tensor = tf.pad(audio_tensor, paddings=padding, mode='CONSTANT', constant_values=0)
    return padded_tensor

def import_audio_from_file(file_path: Path, target_samplerate: int = 22050, target_length_after_resample: int = 88200) ->  tf.Tensor:
    """
    TODO: Partition this function so it better fulfills a single responsibility principle.
    Function that takes as input a filepath, and outputs the appropriately lengthed audio, that is resampled at the appropriate samplerate. 
    If multi-channel audio is given as input, it is converted to to single channel.
    Args:
        file_path (Path): Path to a given audio file.
        target_samplerate (int, optional): Samplerate that is intended for all audio. Defaults to 22050.
        target_length_after_resample (int, optional): Target audio length after resampling. Defaults to 88200.

    Returns:
        tf.Tensor: Output audio.
    """
    audio = tfio.audio.AudioIOTensor(file_path.__str__())
    original_samplerate = tf.cast(audio.rate, tf.int64)
    audio_tensor = audio.to_tensor()
    audio_tensor = tf.cast(audio_tensor, tf.float32)
    audio_tensor = average_audio_file(audio_tensor)
    audio_tensor /= tf.reduce_max(tf.abs(audio_tensor))  # Normalize to range [-1.0, 1.0]
    resampled_audio = tfio.audio.resample(
        audio_tensor,
        rate_in=original_samplerate,
        rate_out=target_samplerate
    )
    if resampled_audio.shape[0] < target_length_after_resample:
        resampled_audio = zero_pad_tensor(resampled_audio, length = target_length_after_resample)
    elif resampled_audio.shape[0] > target_length_after_resample:
        resampled_audio = resampled_audio[:target_length_after_resample, :]
    return resampled_audio

def average_audio_file(audio_tensor: tf.Tensor) -> tf.Tensor:
    """Function that averages the audio tensor along into a single channel.

    Args:
        audio_tensor (tf.Tensor): Audio tensor of shape (audio length, channels).

    Returns:
        tf.Tensor: Audio tensor of shape (audio length, 1)
    """
    mean_tensor = tf.math.reduce_mean(audio_tensor, axis = 1, keepdims = True)
    return mean_tensor
    
def one_hot_encode(class_id: int, max_id: int) -> np.array:
    """Function that converts integer class id into a vector representation as I dislike using keras.losses.SparseCategoricalCrossentropy
    and would rather use keras.losses.CategoricalCrossentropy

    Args:
        class_id (int): ID starting at 0 of class (i.e. 1 implies class 1).
        max_id (int): The number of unique IDs.

    Returns:
        tf.Tensor: One hot encoded vector that converts a class ID to a vector of length = number of classes, with the index of the class entry equal to 1.
    """
    vector_representation = np.zeros((max_id + 1, 1))
    vector_representation[class_id, 0] = 1
    return vector_representation.tolist()

def random_batch_df(df: pd.DataFrame, batch_size: int, random_seed: int = 42) -> pd.DataFrame:
    """Function that batches and resets the index of a pandas data frame.

    Args:
        df (pd.DataFrame): Input dataframe.
        batch_size (int): Size of each batch. 
        random_seed (int, optional): Random seed for sampling. Defaults to 42.

    Returns:
        pd.DataFrame: DataFrame with new column labelled "batch".
    """
    df = df.sample(frac=1, random_state=random_seed)
    df = df.reset_index()
    df['batch'] = df.index // batch_size
    return df
