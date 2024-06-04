from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
import tfkan

#TODO allow for a config or some parameters to be passed to this, so class does not need to be edited directly
class ConvolutionBlock(keras.layers.Layer):

    """Class that creates convolution block of Audio Classification model. Fed into wrapper below that compiles complete model into kersas.Model class.

    Args:
        keras.layers.Layer (): Inherits from a keras layer class, to allow block to be called as a single layer.
    """
    def __init__(self):
        """

        Args:
            num_classes (int): Number of output classes
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

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.max_pooling_1(x)
        x = self.conv_2(x)
        x = self.max_pooling_2(x)
        x = self.conv_3(x)
        x = self.max_pooling_3(x)
        return x

#TODO allow for a config or some parameters to be passed to this, so class does not need to be edited directly
class DenseBlock(keras.layers.Layer):
    """Class that creatses dense block of image classifier. Fed into wrapper below that compiles complete model into kersas.models.Model class.

    Args:
        keras.layers.Layer (): Inherits from a keras layer class, to allow block to be called as a single layer.
    """
    def __init__(self):
        super().__init__()
        self.layer_1 = keras.layers.Dense(128, activation='relu')
        self.layer_2 = keras.layers.Dense(86, activation = 'relu')
        self.dropout_1 = keras.layers.Dropout(0.2)
        self.dropout_2 = keras.layers.Dropout(0.2)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        x = self.dropout_2(x)
        return x
    
class FreqAttentionBlock(keras.layers.Layer):
    """Class that creates our frequency portion of the temporal frequency attention mechanism block using mel spectrograms in a similar manner to Mu., W, et al. 2021
       Paper linked here: https://www.nature.com/articles/s41598-021-01045-4#auth-Wenjie-Mu-Aff1
    Args:
        keras.layers.Layer (): Inherits from a keras layer class, to allow block to be called as a single layer.
    """
    def __init__(self):
        
        super().__init__()
        self.freq_conv_1 = keras.layers.Conv2D(
            64, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_1 = keras.layers.MaxPooling2D(pool_size = (2,1))
        self.freq_conv_2 = keras.layers.Conv2D(
            64, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_2 = keras.layers.MaxPooling2D(pool_size = (2,1))
        self.freq_conv_3 = keras.layers.Conv2D(
            64, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_3 = keras.layers.MaxPooling2D(pool_size = (2,1))
        self.freq_conv_4 = keras.layers.Conv2D(
            64, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_4 = keras.layers.MaxPooling2D(pool_size = (2,1))
        self.freq_conv_5 = keras.layers.Conv2D(
            64, 
            (4, 1), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_5 = keras.layers.MaxPooling2D(pool_size = (2,1))

    def call(self, inputs):
        """_summary_

        Args:
            inputs (_type_): Mel Spectrogram with dimension added so it is 4d

        Returns:
            tf.Tensor: Output frequency attention filter
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
        
        super().__init__()
        self.temp_conv_1 = keras.layers.Conv2D(
            64, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_1 = keras.layers.MaxPooling2D(pool_size = (1,2))
        self.temp_conv_2 = keras.layers.Conv2D(
            64, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_2 = keras.layers.MaxPooling2D(pool_size = (1,2))
        self.temp_conv_3 = keras.layers.Conv2D(
            64, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_3 = keras.layers.MaxPooling2D(pool_size = (1,2))
        self.temp_conv_4 = keras.layers.Conv2D(
            64, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_4 = keras.layers.MaxPooling2D(pool_size = (1,2))
        self.temp_conv_5 = keras.layers.Conv2D(
            64, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_5 = keras.layers.MaxPooling2D(pool_size = (1,2))
        self.temp_conv_6 = keras.layers.Conv2D(
            64, 
            (1, 4), 
            strides = (1,1), 
            padding='valid', 
            activation='relu', 
            kernel_initializer = tf.keras.initializers.HeNormal()
            )
        self.max_pooling_6 = keras.layers.MaxPooling2D(pool_size = (1,2))

    def call(self, inputs):
        """_summary_

        Args:
            inputs (_type_): Mel Spectrogram with dimension added so it is 4d

        Returns:
            tf.Tensor: Output temporal attention filter
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
        temp_attn = self.temp_conv_6(temp_attn)
        temp_attn = self.max_pooling_6(temp_attn)
        
        return temp_attn
    
class CompleteAttentionBlock(keras.layers.Layer):
    """Class that creates our temporal portion of the temporal frequency attention mechanism block using mel spectrograms in a similar manner to Mu., W, et al. 2021
       Paper linked here: https://www.nature.com/articles/s41598-021-01045-4#auth-Wenjie-Mu-Aff1
    Args:
        keras.layers.Layer (): Inherits from a keras layer class, to allow block to be called as a single layer.
    """
    def __init__(self,
                 fft_length=1024,
                 sequence_stride=512,
                 window='hamming',
                 sampling_rate=22050,
                 num_mel_bins=128,
                 power_to_db=True,
                ):
        
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
        """_summary_

        Args:
            inputs (_type_): Mel Spectrogram with dimension added so it is 4d

        Returns:
            tf.Tensor: Output frequency attention filter
        """
        spect = self.mel_spectrogram(inputs)
        spect = self.batch_normalization_input(spect)
        spect = tf.expand_dims(spect, axis = -1)
        freq_filter = self.freq_filter(spect)
        freq_filter = self.batch_normalization_freq(freq_filter)
        temp_filter = self.temp_filter(spect)
        temp_filter = self.batch_normalization_temp(temp_filter)
        # apply filters to spectrogram
        freq_spect = self.multiply_freq([spect, freq_filter])
        temp_spect = self.multiply_temp([spect, temp_filter])
        

        return freq_spect, temp_spect
    
class TempFreqAudioClassificationModel(keras.Model):
    """Class that wraps all relevant blocks to form Temporal-Frequency Attention Based Conv Net

    """

    def __init__(self, num_classes: int, fft_length=1024,
                 sequence_stride=512,
                 window='hamming',
                 sampling_rate=22050,
                 num_mel_bins=128,
                 power_to_db=True,):
        """
        Args:
            num_classes (int): Number of classes a classifier aims to classify.
        """
        super().__init__()
        self.TF_attn_block = CompleteAttentionBlock(fft_length=fft_length,
                                                            sequence_stride=sequence_stride,
                                                            window=window,
                                                            sampling_rate=sampling_rate,
                                                            num_mel_bins = num_mel_bins,
                                                            power_to_db = power_to_db)
        self.average_layer = keras.layers.Average()
        self.conv_block = ConvolutionBlock()
        self.avg_pooling_layer = keras.layers.GlobalAveragePooling2D()
        self.dense_block = DenseBlock()
        self.classifier = keras.layers.Dense(num_classes, activation = 'softmax')

    def call(self, inputs):
        x_1, x_2 = self.TF_attn_block(inputs)
        x = self.average_layer([x_1, x_2])
        x = self.conv_block(x)
        x = self.avg_pooling_layer(x)
        x = self.dense_block(x)
        return self.classifier(x)