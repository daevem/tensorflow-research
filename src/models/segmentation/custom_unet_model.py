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
    concatenate,
)
from tensorflow.keras.models import Model
from models import BaseModel


class CustomUnetModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def create_optimizer(self, optimizer="sgd"):
        super().create_optimizer(optimizer=optimizer)

    def compile(self, loss="binary_crossentropy"):
        self.model.compile(
            optimizer=self.optimizer, loss=loss, metrics=[iou, iou_thresholded],
        )

    def create_model(self):
        input_shape = self.config.input_shape
        self.model = self.custom_unet(input_shape=input_shape)

    def upsample_conv(self, filters, kernel_size, strides, padding):
        return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

    def upsample_simple(self, filters, kernel_size, strides, padding):
        return UpSampling2D(strides)

    def custom_unet(
        self,
        input_shape,
        num_classes=1,
        activation="relu",
        use_batch_norm=True,
        upsample_mode="deconv",  # 'deconv' or 'simple'
        dropout=0.3,
        dropout_change_per_layer=0.0,
        dropout_type="spatial",
        use_dropout_on_upsampling=False,
        filters=16,
        num_layers=4,
        output_activation="sigmoid",
    ):  # 'sigmoid' or 'softmax'

        """
        Customisable UNet architecture (Ronneberger et al. 2015 [1]).
        Arguments:
        input_shape: 3D Tensor of shape (x, y, num_channels)
        num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
        activation (str): A keras.activations.Activation to use. ReLu by default.
        use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
        upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
        dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
        dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
        dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
        use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
        filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
        num_layers (int): Number of total layers in the encoder not including the bottleneck layer
        output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
        Returns:
        model (keras.models.Model): The built U-Net
        Raises:
        ValueError: If dropout_type is not one of "spatial" or "standard"
        [1]: https://arxiv.org/abs/1505.04597
        [2]: https://arxiv.org/pdf/1411.4280.pdf
        """

        if upsample_mode == "deconv":
            upsample = self.upsample_conv
        else:
            upsample = self.upsample_simple

        # Build U-Net model
        inputs = Input(input_shape, name=self.input_name)
        x = inputs

        down_layers = []
        for l in range(num_layers):
            x = conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )
            down_layers.append(x)
            x = MaxPooling2D((2, 2))(x)
            dropout += dropout_change_per_layer
            filters = filters * 2  # double the number of filters with each layer

        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

        if not use_dropout_on_upsampling:
            dropout = 0.0
            dropout_change_per_layer = 0.0

        for conv in reversed(down_layers):
            filters //= 2  # decreasing number of filters with each layer
            dropout -= dropout_change_per_layer
            x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            x = concatenate([x, conv])
            x = conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )

        outputs = Conv2D(
            num_classes, (1, 1), activation=output_activation, name=self.output_name
        )(x)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model


def iou(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth
    )


def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.0):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth
    )


def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
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

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c

