import tensorflow as tf
import numpy as np


class SparseMeanIoU(tf.metrics.MeanIoU):

    def __init__(self, num_classes, name=None, dtype=None):
        """Creates a `SparseMeanIoU` instance.
        Args:
          num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super().__init__(num_classes, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.keras.backend.argmax(y_pred, axis=-1)
        super(SparseMeanIoU, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        super(SparseMeanIoU, self).result()

    def reset_states(self):
        super(SparseMeanIoU, self).reset_states()
