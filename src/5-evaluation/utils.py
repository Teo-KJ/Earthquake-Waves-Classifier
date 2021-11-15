import random

import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence


# Created by DAVIDS
# https://www.kaggle.com/davids1992/specaugment-quick-implementation
def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.15):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num, _ = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec


class DataGenerator(Sequence):
    """Keras data generator
    Sequence-based data generator. Suitable for training and prediction.
    """
    def __init__(self, list_pths, list_labels,
                 to_fit=True, batch_size=32, dim=(100, 150),
                 n_channels=3, n_classes=3, shuffle=True,
                 spec_aug_prob=0.0):
        """Initialization
        :param list_pths: list of all imgs pths to use in the generator
        :param list_labels: list of image labels

        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_pths = list_pths
        self.list_labels = list_labels

        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.spec_aug_prob = spec_aug_prob
        self.on_epoch_end()

    def __len__(self):
        """Denotes no. of batches per epoch
        :return: no. of batches per epoch
        """
        return int(np.floor(len(self.list_pths) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Get batch
        list_pths_temp = [self.list_pths[k] for k in indexes]
        list_labels_temp = [self.list_labels[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_pths_temp)

        if self.to_fit:
            y = self._generate_y(list_labels_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_pths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_pths_temp):
        """Generates data containing batch_size images
        :param list_pths_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, pth in enumerate(list_pths_temp):
            # Augment w probabilty == self.spec_aug_prob
            if random.random() <= self.spec_aug_prob: # augment
                img = self._load_image(pth)
                aug_img = spec_augment(img)
                X[i,] = aug_img
            else: # do not augment
                X[i,] = self._load_image(pth)

        return X

    def _generate_y(self, list_labels_temp):
        """Generates data containing batch_size masks
        :param list_pths_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, label in enumerate(list_labels_temp):
            # Store sample
            y[i,] = label

        return y

    def _load_image(self, image_path):
        """Load image
        :param image_path: path to image to load
        :return: loaded image
        """
        temp_img = Image.open(image_path+'.png') # read in image
        img = temp_img.copy()
        img = np.asarray(img, dtype=np.uint8)
        temp_img.close()
        img = img / 255.0 # scale intensity to between 0 and 1
        return img
