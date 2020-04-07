from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import randint, random, seed
import numpy as np


class SegmentationDataGenerator(Sequence):

    def __init__(self, image_generator_flow, mask_generator_flow, slicing_rate=1.0, slice_sizes=None):
        self.gen_x = image_generator_flow  # type: ImageDataGenerator.flow_from_directory()
        self.gen_y = mask_generator_flow
        if slice_sizes is not None:
            self.sizes = slice_sizes
        else:
            self.sizes = [256, 512]  # , 1024]  # TODO calculate basing on image width and height

        self.slicing_rate = slicing_rate

    def __getitem__(self, index):
        x, y = next(self.gen_x), next(self.gen_y)
        r = random()
        if r < self.slicing_rate:
            r = random()
            if r < 0.5:
                slicing_func = self.vertical_slice
            else:
                slicing_func = self.random_driven_slice
            _x = []
            _y = []
            seed_w = random()
            seed_h = random()
            for i in range(x.shape[0]):
                _xi = x[i]
                _yi = y[i]
                _xi, _yi = slicing_func(_xi, _yi, rate=0.25, seed_w=seed_w, seed_h=seed_h)
                _x.append(_xi)
                _y.append(_yi)
            x = np.array(_x)
            y = np.array(_y)
        return x, y

    def __len__(self):
        return len(self.gen_x)

    def vertical_slice(self, x, y, rate=0.0, seed_w=None, seed_h=None):
        h = x.shape[0]  # shape[0] is height in cv2 notation
        dw = y.shape[1]//4
        offset = randint(0, 4 - 1)
        w = dw * offset

        # do - while
        sliced_mask = y[:, w:w+dw, :].copy()
        while sliced_mask.max == 0:
            offset = randint(0, 4 - 1)
            w = dw * offset
            sliced_mask = y[:, w:w+dw, :].copy()
        sliced_image = x[:, w:w+dw, :].copy()
        return sliced_image, sliced_mask

    def random_driven_slice(self, x, y, rate=0.0, seed_w=None, seed_h=None):
        """
        Returns a sliced version of x and y by randomly choosing among a set of sizes. Given that y is the image mask,
        setting rate param allows to accept all-black image with a certain rate (hence the 'driven').
        Use seed_w and seed_y to generate image batches with same width and height.
        :param x: image
        :param y: image mask
        :param rate: all-black/background-only slice acceptance rate
        :param seed_w: seed the random width size generator
        :param seed_h: seed the random height size generator
        :return: sliced_image, sliced_mask
        """
        def _get_random_slice_params(max_h_segments, max_v_segments, dw, dh):
            offset_w = randint(0, max_h_segments-1)
            w = dw * offset_w
            offset_h = randint(0, max_v_segments-1)
            h = dh * offset_h
            return h, dh, w, dw

        if seed_w is not None:
            seed(seed_w)
        idx = randint(0, len(self.sizes)-1)
        dw = self.sizes[idx]
        dw = min(dw, y.shape[1])

        if seed_h is not None:
            seed(seed_h)
        idx = randint(0, len(self.sizes)-1)
        dh = self.sizes[idx]
        dh = min(dh, y.shape[0])

        horizontal_segments = y.shape[1]//dw
        vertical_segments = y.shape[0]//dh

        seed()  # "clears" seed
        h, dh, w, dw = _get_random_slice_params(horizontal_segments, vertical_segments, dw, dh)

        # do - while
        sliced_mask = y[h:h+dh, w:w+dw, :].copy()
        r = random()
        # in case of totally black image (background only) it will be accepted if random number
        # is lower than the specified rate (i.e. higher rate == higher chance to have black images)
        while sliced_mask.max == 0.0 and r > rate:
            h, dh, w, dw = _get_random_slice_params(horizontal_segments, vertical_segments, dw, dh)
            sliced_mask = y[h:h+dh, w:w+dw, :].copy()
            r = random()

        sliced_image = x[h:h+dh, w:w+dw, :].copy()
        return sliced_image, sliced_mask





