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
from models import BaseModel
import re


class DynamicUnetModel(BaseModel):
    def __init__(self, config):
        self.model = None  # type: tf.keras.models.Model
        super().__init__(config)


    def create_optimizer(self, optimizer="sgd"):
        super().create_optimizer(optimizer=optimizer)

    def compile(self, loss="categorical_crossentropy"):
        self.model.compile(
            optimizer=self.optimizer, loss=loss, metrics=[iou, iou_thresholded],
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
        previous_shape = backbone_layers[0].input_shape[0]
        for c, l in enumerate(backbone_layers):
            if not isinstance(l, MaxPooling2D) and not isinstance(l, Conv2D) and not isinstance(l, Activation):
                continue
            if previous_shape > l.output_shape:
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
        :param backbone_freeze_n: Number of layers to freeze in the backbone model. Use -1 to freeze all (Default -1)
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
            for c, l in enumerate(backbone.layers):
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

        filters = backbone.layers[-1].output_shape[1]  # retrieve number of filters in the last layer of the backbone
        filters *= 2  # and double it

        x = conv2d_block(
            inputs=backbone.output,
            filters=filters,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )

        # for c, l in enumerate(backbone.layers):
        #     print(c, str(l).split(".")[-1].split(" ")[0], l.input_shape, " -> ", l.output_shape)

        idxs = [143, 81, 39, 6, 2]
        for i in idxs:
            filters //= 2  # decreasing number of filters with each layer
            x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            concat_layer = backbone.layers[i-1]  # -1 because we want to use the input to the ith layer as skip link

            if isinstance(concat_layer, ZeroPadding2D):
                concat_layer = Cropping2D(concat_layer.padding)(concat_layer.output)  # remove extra padding
            # setattr(down_layer, "shape", down_layer.input_shape)
            if hasattr(concat_layer, "shape"):
                x = Concatenate()([x, concat_layer])
            else:
                x = Concatenate()([x, concat_layer.output])
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
    dropout=0.0,
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

