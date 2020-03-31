from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class SegmentationDataGenerator(Sequence):

    def __init__(self, image_generator_flow, mask_generator_flow):
        self.gen_x = image_generator_flow  # type: ImageDataGenerator.flow_from_directory()
        self.gen_y = mask_generator_flow

    def __getitem__(self, index):
        return next(self.gen_x), next(self.gen_y)

    def __len__(self):
        return len(self.gen_x)
