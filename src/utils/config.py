import json


class Config:
    def __init__(self, config_path):
        super().__init__()
        with open(config_path) as json_file:
            data = json.load(json_file)
            self.input_shape = data["input_shape"]

            self.train = data["train"]
            self.model = self.train["model"]
            self.train_files_path = self.train["files_path"]
            self.batch_size = self.train["batch_size"]
            self.epochs = self.train["epochs"]
            self.learning_rate = self.train["learning_rate"]
            self.checkpoints_path = self.train["checkpoints_path"]
            self.checkpoint_save_period = self.train["checkpoint_save_period"]
            self.checkpoint_path = self.train["checkpoint_path"]
            self.checkpoint_save_best_only = self.train["checkpoint_save_best_only"]

            self.loss = None
            self.optimizer = None
            self.train_mask_files_path = None
            self.image_data_generator = None
            self.image_data_generator_horizonal_flip = None
            self.image_data_generator_zoom_range = None
            self.image_data_generator_width_shift_range = None
            self.image_data_generator_height_shift_range = None
            self.image_data_generator_rotation_range = None
            self.image_data_generator_featurewise_center = None

            self.test_files_path = None
            self.test_threshold = None

            try:
                self.loss = self.train["loss"]
            except:
                pass

            try:
                self.optimizer = self.train["optimizer"]
            except:
                pass

            try:
                self.train_mask_files_path = self.train["mask_files_path"]
            except:
                pass

            try:
                self.image_data_generator = self.train["image_data_generator"]
                self.image_data_generator_horizonal_flip = self.image_data_generator[
                    "horizontal_flip"
                ]
                self.image_data_generator_zoom_range = self.image_data_generator[
                    "zoom_range"
                ]
                self.image_data_generator_width_shift_range = self.image_data_generator[
                    "width_shift_range"
                ]
                self.image_data_generator_height_shift_range = self.image_data_generator[
                    "height_shift_range"
                ]
                self.image_data_generator_rotation_range = self.image_data_generator[
                    "rotation_range"
                ]
                self.image_data_generator_featurewise_center = self.image_data_generator[
                    "featurewise_center"
                ]
                self.image_data_generator_featurewise_std_normalization = self.image_data_generator[
                    "featurewise_std_normalization"
                ]
            except:
                pass

            try:
                test = data["test"]
                self.test_files_path = test["files_path"]
                try:
                    self.test_threshold = test["test_threshold"]
                except:
                    pass
            except:
                pass