import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """Keras data generator
    Sequence-based data generator. Suitable for training and prediction.
    """
    def __init__(self, list_pths, list_labels,
                 to_fit=True, batch_size=32, dim=(100, 150),
                 n_channels=3, n_classes=3, shuffle=True):
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
            # Store sample
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