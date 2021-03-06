import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Flatten, MaxPooling2D, Reshape)
from tensorflow.keras.models import Model

from models import BaseModel


class ConvClassificationModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)
  
  def create_optimizer(self, optimizer="adam"):
    super().create_optimizer(optimizer=optimizer)

  def compile(self, loss=None):
    super().compile(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

  def plot_predictions(self, test_images):
        for p in self.predictions:
            p_max = p[0].max()
            c_index = p[0].tolist().index(p_max)
            print("{0} | {1}".format(p[0], self.class_names[c_index]))
        layer_outputs = []
        for layer in self.model.layers[:12]:
          layer_outputs.append(layer.output)
        activation_model = Model(self.model.input, layer_outputs)

        for test_image in test_images:
          self.show_activation(test_image, activation_model)
  
  def show_activation(self, test_image, activation_model):
      test_image = np.array(test_image).reshape((1, self.config.input_shape[0], self.config.input_shape[1], self.config.input_shape[2]))
      activations = activation_model.predict(test_image)
      layer_names = []
      for layer in self.model.layers[:12]:
        layer_names.append(layer.name)
      images_per_row = 16
      for layer_name, layer_activation in zip(layer_names, activations):
        features = layer_activation.shape[-1]
        size_w = layer_activation.shape[1]
        size_h = layer_activation.shape[2]
        cols = features // images_per_row
        display_grid = np.zeros((size_w * cols, images_per_row * size_h))
        for col in range(cols):
          for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[col * size_w : (col + 1) * size_w, row * size_h : (row + 1) * size_h] = channel_image
        scale = 1. / size_h
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect="auto", cmap="viridis")
      plt.show()


  def create_model(self):
    input_shape = self.config.input_shape
    inputs = Input(shape=input_shape, name=self.input_name)
    x = inputs
    number_of_classes = 2

    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(16, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(number_of_classes)(x)
    x = Activation('softmax')(x)

    self.model = Model(inputs, x)

  def train(self):
    self.history = self.model.fit(
                self.train_images,
                self.y_train,
                batch_size=self.config.train.batch_size,
                epochs=self.config.train.epochs,
                callbacks=self.callbacks,
                shuffle=True,
                initial_epoch=self.initial_epoch,
                validation_data=(self.validation_images, self.validation_labels)
            )

  def prepare_training(self):
    class_dirs = []
    for di in glob.glob(self.config.train.files_path + "/*"):
      if os.path.isdir(di):
        print("Class: {0}".format(os.path.basename(di)))
        class_dirs.append(di)
    class_count = len(class_dirs)
    print("Detected classes: {0}".format(class_count))
    class_images = []
    class_index = 0
    self.class_names = []
    for di in class_dirs:
      class_name = os.path.basename(di)
      self.class_names.append(class_name)
      images = self.load_images(di, self.config.input_shape)
      class_images.append(dict(index=class_index, name=class_name, images=images))
      class_index += 1
    
    all_images = []
    all_classes = []
    for d in class_images:
      for img in d["images"]:
        all_images.append(img)
        all_classes.append(d["index"])
    all_classes = np.array(all_classes, dtype=np.uint8)
    all_images = np.array(all_images, dtype=np.float32)
    train_data, test_data, train_labels, test_labels = train_test_split(all_images, all_classes, test_size=self.config.train.validation_split, random_state=33)
    self.train_images = train_data
    self.y_train = train_labels
    self.validation_images = test_data
    self.validation_labels = test_labels
