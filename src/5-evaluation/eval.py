'''
This script contains the Seismic() class, which can be used to run regression or classification CNNs. See the bottom of the script for examples of how to call the script for each case. Each type of CNN has the option to either use the full image dataset in your directory, or take a specified number of random samples of the images to run the CNNs. Each type of CNN results in the following outputs:

        Classification:
            * Confusion matrix
            * Confusion matrix plot
            * Accuracy history plot
            * Test accuracy values
        
        Regression:
            * Model loss values
            * Loss history plot
            * Observed vs. predicted output scatterplot

**This script imports images from the directory of images made using "create_images.py", and creates a csv file of metadata. The chunks of data needed to create the metadata file can be downloaded from the STanford EArthquake Dataset (STEAD) here: https://github.com/smousavi05/STEAD.

Please enter user input from lines 50-63. User input requires you to define filepaths to signal data, metadata, and image data as shown below.

Adapated from Kaelynn Rose
on 3/31/2021

'''

import glob
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from model import ResNetCNN
from utils import DataGenerator

keras = tf.keras

################### USER INPUT ####################

data_folder = "../../data/" # root data folder

# path to spectrogram dataset
dir = data_folder + 'eval_images/'

############################### INIT ################################

def get_paths(chunk_id, processed=True):
    if processed:
        csv_pth = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}_processed.csv') # chunk metadata
    else:
        csv_pth = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}.csv') # chunk metadata

    eqpath = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}.hdf5') # chunk earthquake data
    return csv_pth, eqpath

################## END USER INPUT ###################


class SeismicEvaluation():

    def __init__(self, model_type, target, choose_dataset_size, dir):
        self.model_type = model_type
        self.target = target
        self.choose_dataset_size = choose_dataset_size
        self.dir = dir
        self.p_traces_array = []
        self.s_traces_array = []
        self.noise_traces_array = []
        self.img_dataset = []
        self.labels = []
        self.imgs = []
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.model = []
        self.test_loss = []
        self.test_acc = []
        self.predicted_classes = []
        self.predicted_probs = []
        self.cm = []
        self.epochs = []
        self.history = []

        # create list of traces in the image datset
        print('Creating seismic trace list')
        for filename in glob.iglob(os.path.join(dir, 'P/*.png'), recursive=True): # loop through every file in the directory and get trace names from the image files
            self.p_traces_array.append(filename[0:-4]) # remove .png from image file names
        for filename in glob.iglob(os.path.join(dir, 'S/*.png'), recursive=True): # loop through every file in the directory and get trace names from the image files
            self.s_traces_array.append(filename[0:-4]) # remove .png from image file names
        for filename in glob.iglob(os.path.join(dir, 'noise/*.png'), recursive=True): # loop through every file in the directory and get trace names from the image files
            self.noise_traces_array.append(filename[0:-4]) # remove .png from image file names
        print(f'Num noise: {len(self.noise_traces_array)}\nNum P-waves: {len(self.p_traces_array)}\nNum S-waves: {len(self.s_traces_array)}')
        min_wave_type_size = min(len(self.noise_traces_array), len(self.p_traces_array), len(self.s_traces_array))

        if self.model_type == 'classification':
            if choose_dataset_size <= 0:
                choose_dataset_size = min_wave_type_size
            else:
                choose_dataset_size = min(min_wave_type_size, choose_dataset_size)
            print(f'Dataset size: {choose_dataset_size}')
                
            # random choice of images from the directory
            choose_p_dataset = random.sample(self.p_traces_array, choose_dataset_size)
            choose_s_dataset = random.sample(self.s_traces_array, choose_dataset_size)
            choose_noise_dataset = random.sample(self.noise_traces_array, choose_dataset_size)

            self.img_dataset = choose_noise_dataset + choose_p_dataset + choose_s_dataset
            self.labels = np.array([0] * choose_dataset_size + [1] * choose_dataset_size + [2] * choose_dataset_size) # target variable, 'noise' is 0 | 'P' is 1 | 'S' is 2 
        else:
            print('Error: please choose either "classification" or "regression" for CNN model type')

    def evaluate_classification_model(self, model_pth=None, img_save_pth=''):
        if model_pth:
            print('Loading model')
            self.model = tf.keras.models.load_model(model_pth)

            self.val_generator = DataGenerator(
                self.img_dataset, self.labels,
                to_fit=True, batch_size=1, dim=(100, 150),
                n_channels=3, n_classes=3, shuffle=False
            )

        print('Evaluating model on test dataset')
        self.test_loss, self.test_acc = self.model.evaluate_generator(self.val_generator, verbose=0) # get model evaluation metrics
        print("\nTest data, accuracy: {:5.2f}%".format(100*self.test_acc))

        test_images = []
        for i, img_pth in enumerate(self.img_dataset): # loop through trace names in filtered dataframe and append images to imgs array
            temp_img = Image.open(img_pth+'.png') # read in image
            img = temp_img.copy()
            test_images.append(np.asarray(img, dtype=np.uint8))
            temp_img.close()
            sys.stdout.write('\rWorking on trace %04d of %04d' % (i+1, len(self.img_dataset)))
        print()
        self.test_images = np.array(test_images) / 255  

        self.test_labels = self.labels


        print('Finding predicted classes and probabilities to build confusion matrix')
        self.predicted_classes = np.argmax(self.model.predict(self.test_images),axis=-1) # predict the class of each image
        self.predicted_probs = self.model.predict(self.test_images) # predict the probability of each image belonging to a class

        # create confusion matrix
        print('Confusion matrix:')
        self.cm = confusion_matrix(self.test_labels, self.predicted_classes) # compare target values to predicted values and show confusion matrix
        print(self.cm)

        accuracy = accuracy_score(self.test_labels,  self.predicted_classes)
        print(f'\nAccuracy: {accuracy}')

        # Micro average metrics: 
        # Calculate metrics globally by counting the total TPs, FNs & FPs.
        precision = precision_score(self.test_labels, self.predicted_classes, average='micro', labels=np.unique(self.predicted_classes))
        recall = recall_score(self.test_labels, self.predicted_classes, average='micro')
        f1 = f1_score(self.test_labels, self.predicted_classes, average='micro', labels=np.unique(self.predicted_classes))
        print('\n[Micro Average]:')
        print(f'Precision: {precision}\nRecall: {recall}\nF1 score: {f1}')

        # Weighted average metrics: 
        # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label) to account for label imbalance.
        # NOTE: can result in an F-score that is not between precision and recall
        precision = precision_score(self.test_labels, self.predicted_classes, average='weighted')
        recall = recall_score(self.test_labels, self.predicted_classes, average='weighted')
        f1 = f1_score(self.test_labels, self.predicted_classes, average='weighted', labels=np.unique(self.predicted_classes))
        print('\n[weighted Average]:')
        print(f'Precision: {precision}\nRecall: {recall}\nF1 score: {f1}')

        # create img save folder
        Path(img_save_pth).mkdir(parents=True, exist_ok=True)

        # plot confusion matrix
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['Noise', 'P-waves', 'S-waves'])
        disp.plot(cmap='Blues', values_format='')
        plt.title(f'Classification CNN Results')
        plt.tight_layout()
        plt.savefig(os.path.join(img_save_pth, 'confusion_matrix.png'))
        # plt.show()
        
        # plot accuracy history
        # plt.style.use('ggplot')
        # fig, ax = plt.subplots(figsize=(7,7))
        # ax.plot(self.history.history['accuracy'])
        # ax.plot(self.history.history['val_accuracy'])
        # ax.set_title('Model Accuracy')
        # ax.set_ylabel('Accuracy')
        # ax.set_xlabel('Epoch')
        # ax.legend(['train','test'])
        # plt.savefig(os.path.join(img_save_pth, 'model_accuracy.png'))
        # plt.show()

    
# Using the class for a classification CNN
s = SeismicEvaluation(model_type='classification', target='trace_category', choose_dataset_size=-1, dir=dir) # initialize the class

# evaluate the model by inputting model pth
s.evaluate_classification_model(
    model_pth='../3-resnet-model/saved_models/waves_-1dataset_classification_trace_category_epochs15_20211115', 
    img_save_pth='resnet'
) 
