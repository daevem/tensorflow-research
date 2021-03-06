import os
import unittest

from utils.config import Config


class Test_Config(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        self.current_directory = os.getcwd()
        self.config_file_path = self.current_directory + "/config/sample.json"

    def test_init(self):
        config = Config(self.config_file_path)
        self.assertEqual([512, 512, 3], config.input_shape)
        self.assertEqual(40, config.train.batch_size)
        self.assertEqual(100, config.train.epochs)
        self.assertEqual(1e-4, config.train.learning_rate)
        self.assertEqual("PATH_TO_TRAINING_FILES", config.train.files_path)
        self.assertEqual("PATH_TO_TEST_FILE", config.test_file_path)
        self.assertEqual(0.45, config.test_threshold)


if __name__ == "__main__":
    unittest.main()
