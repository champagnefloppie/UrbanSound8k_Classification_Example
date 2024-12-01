from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras

#TODO allow for a config or some parameters to be passed to this, so class does not need to be edited directly
class ConvolutionBlock(keras.layers.Layer):

    """Class that creates convolution block of Audio Classification model. Fed into wrapper below that compiles complete model into kersas.Model class.

    Args:
        keras.layers.Layer (): Inherits from a keras layer class, to allow block to be called as a single layer.
    """
    def __init__(self):
        """
        Method to initialize the layer block.
        """
        super().__init__()
        self.conv_1 = keras.layers.Conv2D(
            16, 
            (4,4), 
            strides = (2,2), 
            padding='same', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_1 = keras.layers.MaxPooling2D()
        self.conv_2 = keras.layers.Conv2D(
            32, 
            (4,4), 
            strides = (2,2), 
            padding='same', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_2 = keras.layers.MaxPooling2D()
        self.conv_3 = keras.layers.Conv2D(
            64, 
            (4,4), 
            strides = (2,2), 
            padding='same', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_3 = keras.layers.MaxPooling2D()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Layer call.

        Args:
            inputs (tf.Tensor): Input features.

        Returns:
            tf.Tensor: Output feature maps.
        """
        x = self.conv_1(inputs)
        x = self.max_pooling_1(x)
        x = self.conv_2(x)
        x = self.max_pooling_2(x)
        x = self.conv_3(x)
        x = self.max_pooling_3(x)
        return x

#TODO allow for a config or some parameters to be passed to this, so class does not need to be edited directly
class DenseBlock(keras.layers.Layer):
    """Class that creates dense block of image classifier. Fed into wrapper below that compiles complete model into kersas.models.Model class.
    I down sample by a roughly a factor of sqrt(2) at each layer. Dropout is standardized at 0.2. 
    TODO: add drop out as a parameter to the init so it can be included in an optimization schema.

    Args:
        keras.layers.Layer (): Inherits from a keras layer class, to allow block to be called as a single layer.
    """
    def __init__(self):
        """
        Method to initialize the layer block.
        """
        super().__init__()
        self.layer_1 = keras.layers.Dense(128, activation='relu')
        self.layer_2 = keras.layers.Dense(91, activation = 'relu')
        self.layer_3 = keras.layers.Dense(64, activation = 'relu')
        self.dropout_1 = keras.layers.Dropout(0.2)
        self.dropout_2 = keras.layers.Dropout(0.2)
        self.dropout_3 = keras.layers.Dropout(0.2)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call the layer on a given input tensor to get outputs.

        Args:
            inputs (tf.Tensor): Input vector. Check dimensionality, it should be (batch size, features).

        Returns:
            tf.Tensor: Output vector.
        """
        x = self.layer_1(inputs)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        x = self.dropout_2(x)
        x = self.layer_3(x)
        x = self.dropout_3(x)
        return x
    
class FreqAttentionBlock(keras.layers.Layer):
    """Class that creates our frequency portion of the temporal frequency attention mechanism block using mel spectrograms in a similar manner to Mu., W, et al. 2021
       Paper linked here: https://www.nature.com/articles/s41598-021-01045-4#auth-Wenjie-Mu-Aff1
    Args:
        keras.layers.Layer (): Inherits from a keras layer class, to allow block to be called as a single layer.
    """
    def __init__(self):
        """
        Method to initialize the layer block.
        """
        super().__init__()
        self.freq_conv_1 = keras.layers.Conv2D(
            128, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_1 = keras.layers.MaxPooling2D(pool_size = (2,1))
        self.freq_conv_2 = keras.layers.Conv2D(
            128, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_2 = keras.layers.MaxPooling2D(pool_size = (2,1))
        self.freq_conv_3 = keras.layers.Conv2D(
            128, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_3 = keras.layers.MaxPooling2D(pool_size = (2,1))
        self.freq_conv_4 = keras.layers.Conv2D(
            128, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_4 = keras.layers.MaxPooling2D(pool_size = (2,1))
        self.freq_conv_5 = keras.layers.Conv2D(
            128, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_5 = keras.layers.MaxPooling2D(pool_size = (2,1))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Layer call.

        Args:
            inputs (tf.Tensor): Mel Spectrogram with dimension added (Thus it is 4-D)

        Returns:
            tf.Tensor: Output frequency attention filter. This is a "vertical" tensor.
        """
        freq_attn = self.freq_conv_1(inputs)
        freq_attn = self.max_pooling_1(freq_attn)
        freq_attn = self.freq_conv_2(freq_attn)
        freq_attn = self.max_pooling_2(freq_attn)
        freq_attn = self.freq_conv_3(freq_attn)
        freq_attn = self.max_pooling_3(freq_attn)
        freq_attn = self.freq_conv_4(freq_attn)
        freq_attn = self.max_pooling_4(freq_attn)
        freq_attn = self.freq_conv_5(freq_attn)
        freq_attn = self.max_pooling_5(freq_attn)

        return freq_attn
    
class TempAttentionBlock(keras.layers.Layer):
    """Class that creates our temporal portion of the temporal frequency attention mechanism block using mel spectrograms in a similar manner to Mu., W, et al. 2021
       Paper linked here: https://www.nature.com/articles/s41598-021-01045-4#auth-Wenjie-Mu-Aff1
    Args:
        keras.layers.Layer (): Inherits from a keras layer class, to allow block to be called as a single layer.
    """
    def __init__(self):
        """
        Method to initialize the layer block.
        """
        super().__init__()
        self.temp_conv_1 = keras.layers.Conv2D(
            128, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_1 = keras.layers.MaxPooling2D(pool_size = (1,2))
        self.temp_conv_2 = keras.layers.Conv2D(
            128, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_2 = keras.layers.MaxPooling2D(pool_size = (1,2))
        self.temp_conv_3 = keras.layers.Conv2D(
            128, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_3 = keras.layers.MaxPooling2D(pool_size = (1,2))
        self.temp_conv_4 = keras.layers.Conv2D(
            128, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_4 = keras.layers.MaxPooling2D(pool_size = (1,2))
        self.temp_conv_5 = keras.layers.Conv2D(
            128, 
            (1, 2), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_5 = keras.layers.MaxPooling2D(pool_size = (1,6))
        self.temp_conv_6 = keras.layers.Conv2D(
            128, 
            (1, 2), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_6 = keras.layers.MaxPooling2D(pool_size = (1,2))

    def call(self, inputs):
        """Layer call.

        Args:
            inputs (tf.Tensor): Mel Spectrogram with dimension added (Thus it is 4-D)

        Returns:
            tf.Tensor: Output frequency attention filter. This is a "horizontal" tensor.
        """
        temp_attn = self.temp_conv_1(inputs)
        temp_attn = self.max_pooling_1(temp_attn)
        temp_attn = self.temp_conv_2(temp_attn)
        temp_attn = self.max_pooling_2(temp_attn)
        temp_attn = self.temp_conv_3(temp_attn)
        temp_attn = self.max_pooling_3(temp_attn)
        temp_attn = self.temp_conv_4(temp_attn)
        temp_attn = self.max_pooling_4(temp_attn)
        temp_attn = self.temp_conv_5(temp_attn)
        temp_attn = self.max_pooling_5(temp_attn)
        # temp_attn = self.temp_conv_6(temp_attn)
        # temp_attn = self.max_pooling_6(temp_attn)
        
        return temp_attn

class CompleteAttentionBlock(keras.layers.Layer):
    """Class that creates our temporal portion of the temporal frequency attention mechanism block using mel spectrograms in a similar manner to Mu., W, et al. 2021
       Paper linked here: https://www.nature.com/articles/s41598-021-01045-4#auth-Wenjie-Mu-Aff1
    Args:
        keras.layers.Layer (): Inherits from a keras layer class, to allow block to be called as a single layer.
    """
    def __init__(self,
                 fft_length: int = 1024,
                 sequence_stride: int = 512,
                 window: str = 'hamming',
                 sampling_rate: int = 22050,
                 num_mel_bins: int = 128,
                 power_to_db: bool = True,
                ):
        """Method to initialize the frequency-attention mechanism.

        Args:
            fft_length (int, optional): Mel-spectrogram parameter. The number of frequency bins considered for each spectrogram. Defaults to 1024.
            sequence_stride (int, optional): Mel-spectrogram parameter. The number of samples between successive STFT columns. Defaults to 512.
            window (str, optional): Mel-spectrogram parameter. Window function applied to each STFT. Defaults to 'hamming'.
            sampling_rate (int, optional): Mel-spectrogram parameter. Sample rate of the audio in hertz. Defaults to 22050.
            num_mel_bins (int, optional): Mel-spectrogram parameter. Number of mel bins to generate when frequencies are re-grouped. Defaults to 128.
            power_to_db (bool, optional): Mel-spectrogram parameter. Convert magnitude of spectrogram values from power to decibels (log scale). Defaults to True.
        """
        super().__init__()
        self.mel_spectrogram = keras.layers.MelSpectrogram(fft_length=fft_length,
                                                            sequence_stride=sequence_stride,
                                                            window=window,
                                                            sampling_rate=sampling_rate,
                                                            num_mel_bins = num_mel_bins,
                                                            power_to_db = power_to_db
                                                            )
        self.batch_normalization_input = keras.layers.BatchNormalization()
        self.freq_filter = FreqAttentionBlock()
        self.temp_filter = TempAttentionBlock()
        self.batch_normalization_freq = keras.layers.BatchNormalization()
        self.batch_normalization_temp = keras.layers.BatchNormalization()
        self.multiply_freq = keras.layers.Multiply()
        self.multiply_temp = keras.layers.Multiply()

    def call(self, inputs):
        """Layer call. Thi

        Args:
            inputs (tf.Tensor): The audio vector (dimension (batch size, audio length)).

        Returns:
            tf.Tensor: Output frequency attention transformed spectrogram and temporal attention transformed spectrogram.
        """
        spect = self.mel_spectrogram(inputs)
        #spect = self.batch_normalization_input(spect)
        spect = tf.expand_dims(spect, axis = -1)
        freq_filter = self.freq_filter(spect)
        freq_filter = self.batch_normalization_freq(freq_filter)
        temp_filter = self.temp_filter(spect)
        temp_filter = self.batch_normalization_temp(temp_filter)
        # apply filters to spectrogram
        freq_spect = self.multiply_freq([spect, freq_filter])
        temp_spect = self.multiply_temp([spect, temp_filter])
        

        return freq_spect, temp_spect