from glob import glob
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio


def zero_pad_tensor(audio_tensor: tf.Tensor, length: int = 88200) -> tf.Tensor:
    # Calculate padding for the first dimension (rows)
    padding = [[0, length - tf.shape(audio_tensor)[0]], [0, 0]]  # No padding for the second dimension

    # Pad the tensor
    padded_tensor = tf.pad(audio_tensor, paddings=padding, mode='CONSTANT', constant_values=0)
    return padded_tensor

def import_audio_from_file(file_path: Path, target_sample_rate: int = 22050) -> Dict[Path, tf.Tensor]:
    """

    Args:
        file_path (Path): _description_

    Returns:
        Dict[Path, tf.Tensor]: _description_
    """

    audio = tfio.audio.AudioIOTensor(file_path.__str__())
    original_samplerate = tf.cast(audio.rate, tf.int64)
    target_samplerate = 22050
    # Extract audio as a tensor
    audio_tensor = audio.to_tensor()
    audio_tensor = tf.cast(audio_tensor, tf.float32)
    audio_tensor /= tf.reduce_max(tf.abs(audio_tensor))  # Normalize to range [-1.0, 1.0]
    # Resample the audio
    resampled_audio = tfio.audio.resample(
        audio_tensor,
        rate_in=original_samplerate,
        rate_out=target_samplerate
    )
    return resampled_audio

def one_hot_encode(class_id: int, max_id: int) -> np.array:
    """Function that converts integer class id into a vector representation as I dislike using keras.losses.SparseCategoricalCrossentropy
    and would rather use keras.losses.CategoricalCrossentropy

    Args:
        max_id (int): _description_

    Returns:
        tf.Tensor: _description_
    """
    vector_representation = np.zeros((max_id + 1, 1))
    vector_representation[class_id, 0] = 1
    return vector_representation.tolist()

def random_batch_df(df: pd.DataFrame, batch_size: int, random_seed: int = 42) -> pd.DataFrame:
    """Function that 

    Args:
        df (pd.DataFrame): _description_
        batch_size (int): _description_
        random_seed (int, optional): _description_. Defaults to 42.

    Returns:
        pd.DataFrame: _description_
    """
    df = df.sample(frac=1, random_state=random_seed)
    df = df.reset_index()
    df['batch'] = df.index // batch_size
    return df
