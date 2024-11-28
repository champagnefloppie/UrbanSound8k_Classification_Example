from typing import List
from pathlib import Path


import numpy as np
import tensorflow as tf
from tensorflow import keras

class TrainingInstance:
    """Class that tracks a model through its training process
    """
    def __init__(self,
                 model: keras.Model,
                 loss_func: keras.losses.Loss,
                 optimizer: keras.optimizers.Optimizer,
                 training_metrics: List[keras.metrics.Metric],
                 validation_metrics: List[keras.metrics.Metric],
                 ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.training_metrics = training_metrics
        self.validation_metrics = validation_metrics

    @tf.function
    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Function that executes a training step

        Args:
            x (tf.Tensor): _description_
            y (tf.Tensor): _description_

        Returns:
            tf.Tensor: _description_
        """
        with tf.GradientTape() as tape:
            y_hat = self.model(x, training=True)
            loss_value = self.loss_func(y, y_hat)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        for ind, _ in enumerate(self.training_metrics):
            self.training_metrics[ind].update_state(y, y_hat)
        return loss_value
    
    @tf.function
    def test_step(self, x: tf.Tensor, y: tf.Tensor):
        y_hat = self.model(x, training=False)
        for ind, _ in enumerate(self.validation_metrics):
            self.validation_metrics[ind].update_state(y, y_hat)

    def get_train_metric_state(self) -> List[tf.Tensor]:
        current_state = [metric.result() for metric in self.training_metrics]
        return current_state

    def get_validation_metric_state(self) -> List[tf.Tensor]:
        current_state = [metric.result() for metric in self.validation_metrics]
        return current_state
    
    def reset_training_metrics(self):
        for metric in self.training_metrics:
            metric.reset_state()

    def reset_validation_metrics(self):
        for metric in self.validation_metrics:
            metric.reset_state()


    

        


