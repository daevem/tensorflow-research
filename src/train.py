import argparse
import os
import random
import sys
import cv2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from utils.config import Config
from utils.model_creater import create_model


def main():
    parser = argparse.ArgumentParser(description="Used to train TensorFlow model")
    parser.add_argument(
        "config",
        metavar="config",
        help="Path to the configuration file containing all parameters for model training",
    )
    parser.add_argument(
        "--train_files_path",
        dest="train_files_path",
        metavar="path",
        help="Overwrites the path included in the config to the training files",
    )
    parser.add_argument(
        "--train_mask_files_path",
        dest="train_mask_files_path",
        metavar="path",
        help="Overwrites the path included in the config to the training mask files",
    )
    parser.add_argument(
        "--test_file_path",
        dest="test_file_path",
        metavar="path",
        help="Overwrites the path included in the config to the test file",
    )
    parser.add_argument(
        "--test_threshold",
        dest="test_threshold",
        metavar="number (0-1)",
        type=float,
        help="Overwrites the test threshold",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        metavar="number (1-n)",
        type=int,
        help="Overwrites the train epochs",
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        metavar="string (e.g. 'mse')",
        help="Overwrites the train loss parameter",
    )
    parser.add_argument(
        "--model",
        dest="model",
        metavar="string (e.g. 'small', 'advanced', 'small_unet')",
        help="Overwrites the train model",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        metavar="number (1-n)",
        type=int,
        help="Overwrites the train batch size",
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        metavar="number",
        type=float,
        help="Overwrites the learning rate",
    )
    parser.add_argument(
        "--checkpoint_path",
        dest="checkpoint_path",
        metavar="path",
        help="Overwrites the path to the saved checkpoint containing the model weights",
    )
    parser.add_argument(
        "--plot_history",
        dest="plot_history",
        metavar="boolean (default: false)",
        type=bool,
        help="Plots the model training history",
    )

    args = parser.parse_args()
    config_path = args.config
    config = Config(config_path)
    plot_history = False

    # Overwrite config
    if args.train_files_path:
        config.train.files_path = args.train_files_path
    if args.train_mask_files_path:
        config.train.mask_files_path = args.train_mask_files_path
    if args.test_file_path:
        config.test_file_path = args.test_file_path
    if args.test_threshold:
        config.test_threshold = args.test_threshold
    if args.epochs:
        config.train.epochs = args.epochs
    if args.loss:
        config.train.loss = args.loss
    if args.model:
        config.model = args.model
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.learning_rate:
        config.train.learning_rate = args.learning_rate
    if args.checkpoint_path:
        config.train.checkpoint_path = args.checkpoint_path

    if args.plot_history:
        plot_history = True

    # Set seed to get reproducable experiments
    seed_value = 33
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # ToDo: Create model
    model = create_model(config)  # type: BaseModel
    model.summary()

    # ToDo: Train model
    model.train()
    history = model.history
    if history is not None:
        epochs = len(history.epoch) + model.initial_epoch
        # model.model.save_weights(
        #     config.train.checkpoints_path + "/model-{0:04d}.ckpts".format(epochs)
        # )
        model.save("resnet50_unet_vertical_and_random_slices_catBNrelu_freeze0_featurewise.h5")
        if plot_history:
            model.plot_history()
    # model.model = tf.keras.models.load_model("resnet50_unet_vertical_and_random_slices_catBNrelu.h5")
    val_path = os.path.join(config.train.val_files_path, config.train.val_classes[0])
    val_masks_path = os.path.join(config.train.val_masks_path, config.train.val_classes[0])
    images = os.listdir(val_path)
    random.shuffle(images)
    images = images[:10]
    test_images = []
    for f in images:
        test_images.append(cv2.imread(os.path.join(val_path, f)))
    mask_files = ["".join(f.split(".")[0]) + "_gray.png" for f in images]
    gray_masks = []
    for f in mask_files:
        gray_masks.append(cv2.imread(os.path.join(val_masks_path, f), cv2.IMREAD_GRAYSCALE))
    color_masks = []
    color_map = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255]
    }
    for m in gray_masks:
        cm = get_colorized_map(m, color_map)
        color_masks.append(cm)

    predictions = np.array(model.predict(test_images))

    amax = np.argmax(predictions, axis=-1)
    colorized_predictions = []
    for pred in amax:
        cm = get_colorized_map(pred, color_map)
        colorized_predictions.append(cm)

    for mask_name, gt, pred in zip(mask_files, color_masks, colorized_predictions):
        base_name = mask_name.strip("_gray.png")
        mask_name = base_name + "_color_gt.png"
        pred_name = base_name + "_color_pred.png"
        path_to_pred = os.path.join("predictions", "resnet50_unet_vertical_and_random_slices_catBNrelu")
        os.makedirs(path_to_pred, exist_ok=True)
        cv2.imwrite(os.path.join(path_to_pred, mask_name), gt)
        cv2.imwrite(os.path.join(path_to_pred, pred_name), pred)

    print("Training completed, predictions built.")
    K.clear_session()


def get_colorized_map(grayscale_image, color_map):
    m = grayscale_image
    color_mask = np.zeros(m.shape + (3,), dtype=np.uint8)
    color_mask[m == 0] = color_map[0]
    color_mask[m == 1] = color_map[1]
    color_mask[m == 2] = color_map[2]
    color_mask[m == 3] = color_map[3]
    return color_mask


if __name__ == "__main__":
    main()
