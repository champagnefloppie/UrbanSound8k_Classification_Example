import tensorflow as tf
from tensorflow import keras

import model_training.models.layer_blocks as lb

# TODO: Decouple this model from the layers within it. Pass the layers to it. Would be a more sound way of doing this. 
# NOTE: This model is designed to take as input 4 seconds of audio sampled at 22050Hz (array of shape (batch_size, 88200))

class TempFreqAudioClassificationModel(keras.Model):
    """
    Class that wraps/constructs all relevant blocks to form Temporal-Frequency Attention Based Conv Net.

    """

    def __init__(self, num_classes: int,
                 fft_length: int = 1024,
                 sequence_stride: int = 512,
                 window: str = 'hamming',
                 sampling_rate: int = 22050,
                 num_mel_bins: int = 128,
                 power_to_db: bool = True,):
        """
        Method to initialize the Model. This method collects all the above layers, and converts them into a model instance.
    
        Args:
            num_classes (int): Number of classes a classifier aims to classify. This inform size of last layer of dense network that
                               is appended to the dense block.
            fft_length (int, optional): Mel-spectrogram parameter, passed to CompleteAttentionBlock. The number of frequency bins considered for each spectrogram. Defaults to 1024.
            sequence_stride (int, optional): Mel-spectrogram parameter, passed to CompleteAttentionBlock. The number of samples between successive STFT columns. Defaults to 512.
            window (str, optional): Mel-spectrogram parameter, passed to CompleteAttentionBlock. Window function applied to each STFT. Defaults to 'hamming'.
            sampling_rate (int, optional): Mel-spectrogram parameter, passed to CompleteAttentionBlock. Sample rate of the audio in hertz. Defaults to 22050.
            num_mel_bins (int, optional): Mel-spectrogram parameter, passed to CompleteAttentionBlock. Number of mel bins to generate when frequencies are re-grouped. Defaults to 128.
            power_to_db (bool, optional): Mel-spectrogram parameter, passed to CompleteAttentionBlock. Convert magnitude of spectrogram values from power to decibels (log scale). Defaults to True.
        """ 
        super().__init__()
        self.TF_attn_block = lb.CompleteAttentionBlock(fft_length=fft_length,
                                                            sequence_stride=sequence_stride,
                                                            window=window,
                                                            sampling_rate=sampling_rate,
                                                            num_mel_bins = num_mel_bins,
                                                            power_to_db = power_to_db)
        self.average_layer = keras.layers.Average()
        self.conv_block = lb.ConvolutionBlock()
        self.avg_pooling_layer = keras.layers.GlobalAveragePooling2D()
        self.flatten_layer = keras.layers.Flatten()
        self.dense_block = lb.DenseBlock()
        self.classifier = keras.layers.Dense(num_classes, activation = 'softmax')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Model call on raw audio input.

        Args:
            inputs (tf.Tensor): Audio in raw form. Must be float32 scaled between -1 and 1, with shape (batch size, 88200).

        Returns:
            tf.Tensor: Estimated class probabilities.
        """
        x_1, x_2 = self.TF_attn_block(inputs)
        x = self.average_layer([x_1, x_2])
        x = self.conv_block(x)
        # x = self.avg_pooling_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_block(x)
        return self.classifier(x)