import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    SpatialDropout2D,
    UpSampling2D,
    Concatenate,
    ZeroPadding2D,
    Cropping2D
)
from tensorflow.keras.models import Model
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
import numpy as np
from models import BaseModel
import re


class DynamicUnetModel(BaseModel):
	def __init__(self, config):
		self.model = None  # type: tf.keras.models.Model
		super().__init__(config)

	def create_optimizer(self, optimizer="adam"):
		super().create_optimizer(optimizer=optimizer)

	def compile(self, loss="categorical_crossentropy"):
		self.model.compile(
		optimizer=self.optimizer, loss=loss,
		metrics=["accuracy", MeanIoU(num_classes=4, name="MeanIoU_noBG")]  # , mean_iou_no_bg],
		)

	def create_model(self):
		if self.config.model_params.backbone == "resnet50":
			backbone = tf.keras.applications.ResNet50(input_shape=self.config.input_shape,
							weights=self.config.model_params.pretrained_on,
							include_top=False)
		else:
			raise ValueError("Unsupported backbone: {}".format(self.config.model_params.backbone))
		self.model = self.dynamic_unet(backbone,
					num_classes=self.config.n_classes,
					upsample_mode=self.config.model_params.upsample_mode,
					backbone_freeze_n=self.config.model_params.freeze_layers)
		return self.model

	def summary(self):
		self.model.summary()

	def upsample_conv(self, filters, kernel_size, strides, padding):
		return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

	def upsample_simple(self, filters, kernel_size, strides, padding):
		return UpSampling2D(strides)

	@staticmethod
	def get_shape_change_idxs(backbone_layers):
		idxs = []
		n_channels = backbone_layers[0].input_shape[0][-1]
		previous_shape = backbone_layers[0].input_shape[0]
		for c, l in enumerate(backbone_layers):
			if not isinstance(l, MaxPooling2D) and not isinstance(l, Conv2D) and not isinstance(l, Activation):
				continue
			if n_channels < l.output_shape[-1]:
				n_channels = l.output_shape[-1]
				previous_shape = l.output_shape
				idxs.append(c)
		return idxs

	def dynamic_unet(
		self,
		backbone: tf.keras.models.Model,
		num_classes=1,
		activation="relu",
		use_batch_norm=True,
		upsample_mode="deconv",  # 'deconv' or 'simple'
		output_activation="softmax",  # 'sigmoid' or 'softmax'
		backbone_freeze_n=-1,
	):

		"""
		Dynamic UNet architecture (Ronneberger et al. 2015 [1]) creator. Given a backbone(encoder) it dynamically
		creates a suitable UNet architecture by appending a decoder.
		Arguments:
		:param backbone: tf.keras.models.Model instance with downscaling rate = none or 2 between different layers
		:param num_classes:  (int)Unique classes in the output mask. Should be set to 1 for binary segmentation
		:param activation: (str) A keras.activations.Activation to use. ReLu by default.
		:param use_batch_norm: (bool) Whether to use Batch Normalisation across the channel axis between convolutional layers
		:param upsample_mode: (one of "deconv" or "simple") Whether to use transposed convolutions or simple upsampling in the decoder part
		:param output_activation: (str) A keras.activations.Activation to use. Sigmoid by default for binary segmentation
		:param backbone_freeze_n: Number of (Conv) layers to freeze in the backbone model. Use -1 to freeze all (Default -1)
		:return: The built U-Net model (tf.keras.models.Model)
		[1]: https://arxiv.org/abs/1505.04597
		[2]: https://arxiv.org/pdf/1411.4280.pdf
		"""

		if upsample_mode == "deconv":
		    upsample = self.upsample_conv
		else:
		    upsample = self.upsample_simple

		# Build U-Net model
		# inputs = Input(input_shape, name=self.input_name)
		# x = inputs

		if backbone_freeze_n > -1:
			for c, l in enumerate([_l for _l in backbone.layers if isinstance(_l, Conv2D)]):
				if c <= backbone_freeze_n:
					l.trainable = False
				else:
					l.trainable = True
		else:
			backbone.trainable = False

		# get indexes of the backbone layers (max pooling or conv2d) in which there is a shape change
		idxs = DynamicUnetModel.get_shape_change_idxs(backbone.layers)
		# reverse their order to use them in encoder-to-decoder skip connections
		idxs = idxs[::-1]
		idxs.pop(0)

		filters = backbone.layers[-1].output_shape[-1]  # retrieve number of filters in the last layer of the backbone
		# filters *= 2  # and double it

		# commented because we are using last resnet block as encoder-to-decoder bridge
		# x = conv2d_block(
		#     inputs=backbone.output,
		#     filters=filters,
		#     use_batch_norm=use_batch_norm,
		#     activation=activation,
		# )
		x = backbone.output
		filters //= 2
		x = BatchNormalization()(x)
		x = Activation("relu")(x)
		x = Conv2D(filters, 1, activation="relu", name="bridge_conv2d")(x)
		x = BatchNormalization()(x)
		# for c, l in enumerate(backbone.layers):
		#     print(c, str(l).split(".")[-1].split(" ")[0], l.input_shape, " -> ", l.output_shape)

		# idxs = [143, 81, 39, 6, 2]  # resnet50 layers
		for i in idxs:
			if filters > 16:  # filter n. lower bound
				filters //= 2  # decreasing number of filters with each layer
			x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
			concat_layer = backbone.layers[i]  # -1 because we want to use the input to the i-th layer as skip link

			if isinstance(concat_layer, ZeroPadding2D):
				concat_layer = Cropping2D(concat_layer.padding)(concat_layer.output)  # remove extra padding
			# setattr(down_layer, "shape", down_layer.input_shape)
			if hasattr(concat_layer, "shape"):
				...
			else:
				concat_layer = concat_layer.output
			concat_layer = BatchNormalization()(concat_layer)
			x = Concatenate()([x, concat_layer])
			x = Activation("relu")(x)
			x = conv2d_block(
				inputs=x,
				filters=filters,
				use_batch_norm=use_batch_norm,
				activation=activation,
			)
		# last up-sampling layer
		filters //= 2
		x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)

		outputs = Conv2D(
		    num_classes, (1, 1), activation=output_activation, name=self.output_name
		)(x)

		model = Model(inputs=[backbone.input], outputs=[outputs])
		return model


def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.0,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="glorot_uniform",
    padding="same",
):

	if dropout_type == "spatial":
		DO = SpatialDropout2D
	elif dropout_type == "standard":
		DO = Dropout
	else:
		raise ValueError(
			f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
		)
	# TODO consider restiling conv+relu+bn to conv+bn+relu
	c = Conv2D(
		filters,
		kernel_size,
		kernel_initializer=kernel_initializer,
		padding=padding,
		use_bias=not use_batch_norm,
	)(inputs)
	if use_batch_norm:
		c = BatchNormalization()(c)
	if dropout > 0.0:
		c = DO(dropout)(c)
	c = Activation(activation)(c)
	c = Conv2D(
		filters,
		kernel_size,
		kernel_initializer=kernel_initializer,
		padding=padding,
		use_bias=not use_batch_norm,
	)(c)
	if use_batch_norm:
		c = BatchNormalization()(c)
	c = Activation(activation)(c)
	return c


class MeanIoU(tf.keras.metrics.Metric):
	"""Computes the mean Intersection-Over-Union metric.

	Mean Intersection-Over-Union is a common evaluation metric for semantic image
	segmentation, which first computes the IOU for each semantic class and then
	computes the average over classes. IOU is defined as follows:
	IOU = true_positive / (true_positive + false_positive + false_negative).
	The predictions are accumulated in a confusion matrix, weighted by
	`sample_weight` and the metric is then calculated from it.

	If `sample_weight` is `None`, weights default to 1.
	Use `sample_weight` of 0 to mask values.

	Usage:

	```python
	m = tf.keras.metrics.MeanIoU(num_classes=2)
	m.update_state([0, 0, 1, 1], [0, 1, 0, 1])

	# cm = [[1, 1],
	    [1, 1]]
	# sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
	# iou = true_positives / (sum_row + sum_col - true_positives))
	# result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
	print('Final result: ', m.result().numpy())  # Final result: 0.33
	```

	Usage with tf.keras API:

	```python
	model = tf.keras.Model(inputs, outputs)
	model.compile(
	'sgd',
	loss='mse',
	metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
	```
	"""

	def __init__(self, num_classes, name=None, dtype=None):
		"""Creates a `MeanIoU` instance.

		Args:
		num_classes: The possible number of labels the prediction task can have.
		This value must be provided, since a confusion matrix of dimension =
		[num_classes, num_classes] will be allocated.
		name: (Optional) string name of the metric instance.
		dtype: (Optional) data type of the metric result.
		"""
		super(MeanIoU, self).__init__(name=name, dtype=dtype)
		self.num_classes = num_classes

		# Variable to accumulate the predictions in the confusion matrix. Setting
		# the type to be `float64` as required by confusion_matrix_ops.
		self.total_cm = self.add_weight(
			'total_confusion_matrix',
			shape=(num_classes, num_classes),
			initializer=init_ops.zeros_initializer,
			dtype=dtypes.float64)

	def update_state(self, y_true, y_pred, sample_weight=None):
		"""Accumulates the confusion matrix statistics.

		Args:
		y_true: The ground truth values.
		y_pred: The predicted values.
		sample_weight: Optional weighting of each example. Defaults to 1. Can be a
		`Tensor` whose rank is either 0, or the same rank as `y_true`, and must
		be broadcastable to `y_true`.

		Returns:
		Update op.
		"""

		y_pred = math_ops.argmax(y_pred, axis=-1)
		y_true = math_ops.cast(y_true, self._dtype)
		y_pred = math_ops.cast(y_pred, self._dtype)

		# Flatten the input if its rank > 1.
		if y_pred.shape.ndims > 1:
			y_pred = array_ops.reshape(y_pred, [-1])

		if y_true.shape.ndims > 1:
			y_true = array_ops.reshape(y_true, [-1])

		if sample_weight is not None and sample_weight.shape.ndims > 1:
			sample_weight = array_ops.reshape(sample_weight, [-1])

		# Accumulate the prediction to current confusion matrix.
		current_cm = confusion_matrix.confusion_matrix(
			y_true,
			y_pred,
			self.num_classes,
			weights=sample_weight,
			dtype=dtypes.float64)
		return self.total_cm.assign_add(current_cm)

	def result(self):
		"""Compute the mean intersection-over-union via the confusion matrix."""
		total_cm = self.total_cm[0:, 0:]
		sum_over_row = math_ops.cast(
			math_ops.reduce_sum(total_cm, axis=0), dtype=self._dtype)
		sum_over_col = math_ops.cast(
			math_ops.reduce_sum(total_cm, axis=1), dtype=self._dtype)
		true_positives = math_ops.cast(
			array_ops.diag_part(total_cm), dtype=self._dtype)

		# sum_over_row + sum_over_col =
		#     2 * true_positives + false_positives + false_negatives.
		denominator = sum_over_row + sum_over_col - true_positives

		# The mean is only computed over classes that appear in the
		# label or prediction tensor. If the denominator is 0, we need to
		# ignore the class.
		num_valid_entries = math_ops.reduce_sum(
			math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

		iou = math_ops.div_no_nan(true_positives, denominator)
		iou = tf.where(tf.constant([False] + ([True]*3)), iou, tf.constant([0.0]*4))

		return math_ops.div_no_nan(
			math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)

	def reset_states(self):
		K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

	def get_config(self):
		config = {'num_classes': self.num_classes}
		base_config = super(MeanIoU, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def iou(y_true, y_pred, label: int):
	"""
	Return the Intersection over Union (IoU) for a given label.
	Args:
	y_true: the expected y values as a one-hot
	y_pred: the predicted y values as a one-hot or softmax output
	label: the label to return the IoU for
	Returns:
	the IoU for the given label
	"""
	# extract the label values using the argmax operator then
	# calculate equality of the predictions and truths to the label
	y_true = K.cast(K.equal(y_true, label), K.floatx())
	y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
	# calculate the |intersection| (AND) of the labels
	intersection = K.sum(y_true * y_pred)
	# calculate the |union| (OR) of the labels
	union = K.sum(y_true) + K.sum(y_pred) - intersection
	# avoid divide by zero - if the union is zero, return 1
	# otherwise, return the intersection over union
	return K.switch(K.equal(union, 0), 1.0, intersection / union)


def build_iou_for(label: int, name: str=None):
	"""
	Build an Intersection over Union (IoU) metric for a label.
	Args:
	label: the label to build the IoU metric for
	name: an optional name for debugging the built method
	Returns:
	a keras metric to evaluate IoU for the given label

	Note:
	label and name support list inputs for multiple labels
	"""
	# handle recursive inputs (e.g. a list of labels and names)
	if isinstance(label, list):
		if isinstance(name, list):
			return [build_iou_for(l, n) for (l, n) in zip(label, name)]
		return [build_iou_for(l) for l in label]

	# build the method for returning the IoU of the given label
	def label_iou(y_true, y_pred):
		"""
		Return the Intersection over Union (IoU) score for {0}.
		Args:
		    y_true: the expected y values as a one-hot
		    y_pred: the predicted y values as a one-hot or softmax output
		Returns:
		    the scalar IoU value for the given label ({0})
		""".format(label)
		return iou(y_true, y_pred, label)

		# if no name is provided, us the label
		if name is None:
			name = label
		# change the name of the method for debugging
		label_iou.__name__ = 'iou_{}'.format(name)

		return label_iou
	
total_iou = None

@tf.function
def mean_iou_no_bg(y_true, y_pred):
	"""
	Return the Intersection over Union (IoU) score.
	Args:
	y_true: the expected y values as a one-hot
	y_pred: the predicted y values as a one-hot or softmax output
	Returns:
	the scalar IoU value (mean over all labels)
	"""
	# get number of labels to calculate IoU for
	num_labels = K.int_shape(y_pred)[-1]
	# initialize a variable to store total IoU in
	total_iou = iou(y_true, y_pred, 1)
	# iterate over labels to calculate IoU for
	for label in range(2, num_labels):
		total_iou = total_iou + iou(y_true, y_pred, label)
	# divide total IoU by number of labels to get mean IoU
	return total_iou / (num_labels-1)