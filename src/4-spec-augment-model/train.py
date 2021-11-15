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
from utils import DataGenerator

from model import ResNetCNN

keras = tf.keras

################### USER INPUT ####################

data_folder = "../../data/" # root data folder

# path to spectrogram dataset
dir = data_folder + 'images/'

############################### INIT ################################

def get_paths(chunk_id, processed=True):
    if processed:
        csv_pth = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}_processed.csv') # chunk metadata
    else:
        csv_pth = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}.csv') # chunk metadata

    eqpath = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}.hdf5') # chunk earthquake data
    return csv_pth, eqpath

# read the noise and earthquake csv files into separate dataframes:
noise = pd.read_csv(get_paths(1, processed=False)[0])
earthquakes_1 = pd.read_csv(get_paths(2, processed=False)[0])
# earthquakes_2 = pd.read_csv(get_paths(3, processed=False)[0])
# earthquakes_3 = pd.read_csv(get_paths(4, processed=False)[0])
# earthquakes_4 = pd.read_csv('get_paths(5, processed=False)[0])
# earthquakes_5 = pd.read_csv(get_paths(6, processed=False)[0])

# full_csv = pd.concat([noise, earthquakes_1, earthquakes_2, earthquakes_3, earthquakes_4, earthquakes_5])
full_csv = pd.concat([noise, earthquakes_1])

################## END USER INPUT ###################


class Seismic():

    def __init__(self, model_type, target, choose_dataset_size, full_csv, dir):
        self.model_type = model_type
        self.target = target
        self.choose_dataset_size = choose_dataset_size
        self.full_csv = full_csv
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
            # len_img_dataset = len(self.img_dataset)
            # print(f'The number of traces in the directory is {len_img_dataset}')
            # for i, img_pth in enumerate(self.img_dataset): # loop through trace names in filtered dataframe and append images to imgs array
            #     temp_img = Image.open(img_pth+'.png') # read in image
            #     img = temp_img.copy()
            #     self.imgs.append(np.asarray(img, dtype=np.uint8))
            #     temp_img.close()
            #     sys.stdout.write('\rWorking on trace %04d of %04d' % (i+1, len_img_dataset))
            # print()
            # self.imgs = np.array(self.imgs)   
        else:
            print('Error: please choose either "classification" or "regression" for CNN model type')
 

    def train_test_split(self, test_size, random_state):
        # train test split using sklearn
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self.img_dataset, self.labels, random_state=random_state, test_size=test_size)
        print(f'Train shape:\t{len(self.train_images)}')
        print(f'Train labels shape:\t{len(self.train_labels)}')
        print(f'Test images shape:\t{len(self.test_images)}')
        print(f'Test labels shape:\t{len(self.test_labels)}')

        self.train_generator = DataGenerator(
            self.train_images, self.train_labels,
            to_fit=True, batch_size=32, dim=(100, 150),
            n_channels=3, n_classes=3, shuffle=True,
            spec_aug_prob=0.5
        )
        self.val_generator = DataGenerator(
            self.test_images, self.test_labels,
            to_fit=True, batch_size=1, dim=(100, 150),
            n_channels=3, n_classes=3, shuffle=False
        )
        
        # print('Scaling image intensity')
        # self.train_images = self.train_images/255.0 # scale intensity to between 0 and 1
        # self.test_images = self.test_images/255.0 # scale intensity to between 0 and 1

        # img_height = self.train_images.shape[1] # get height of each image in pixels
        # img_width = self.train_images.shape[2] # get width of each image in pixels

        # print('Resizing images')
        # self.train_images = self.train_images.reshape(-1,img_height,img_width,3) # reshape to input into CNN which requires a 4D-tensor
        # self.test_images = self.test_images.reshape(-1,img_height,img_width,3) # reshape to input into CNN which requires a 4D-tensor

    def classification_cnn(self, epochs):
        self.epochs = epochs
        
        # set callbacks so that the model will be saved after each epoch
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f'./saved_models/waves_{str(self.choose_dataset_size)}dataset_{self.model_type}_{self.target}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}',
                save_freq='epoch')
        ]

        # build CNN on dataset
        print('Building CNN model')
        img_shape = (100, 150, 3)
        model = ResNetCNN().create_model(img_shape)

        # self.history = model.fit(
        #     self.train_images, self.train_labels, 
        #     epochs=epochs, callbacks=callbacks, 
        #     validation_split=0.2
        # ) # fit model and save history

        self.history = model.fit_generator(
            generator=self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs, callbacks=callbacks, 
            use_multiprocessing=True, workers=8,
        ) # fit model and save history

        print(model.summary())
        
        # Set model save path
        self.saved_model_path = f'./saved_models/waves_{str(self.choose_dataset_size)}dataset_{self.model_type}_{self.target}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}'
        # Save entire model to a HDF5 file
        model.save(self.saved_model_path)
        self.model = model

    def evaluate_classification_model(self):
        print('Evaluating model on test dataset')
        self.test_loss, self.test_acc = self.model.evaluate_generator(self.val_generator, verbose=0) # get model evaluation metrics
        print("\nTest data, accuracy: {:5.2f}%".format(100*self.test_acc))

        test_images = []
        for i, img_pth in enumerate(self.test_images): # loop through trace names in filtered dataframe and append images to imgs array
            temp_img = Image.open(img_pth+'.png') # read in image
            img = temp_img.copy()
            test_images.append(np.asarray(img, dtype=np.uint8))
            temp_img.close()
            sys.stdout.write('\rWorking on trace %04d of %04d' % (i+1, len(self.test_images)))
        print()
        self.test_images = np.array(test_images) / 255  


        print('Finding predicted classes and probabilities to build confusion matrix')
        self.predicted_classes = np.argmax(self.model.predict(self.test_images),axis=-1) # predict the class of each image
        self.predicted_probs = self.model.predict(self.test_images) # predict the probability of each image belonging to a class

        # create confusion matrix
        print('Confusion matrix:')
        self.cm = confusion_matrix(self.test_labels,self.predicted_classes) # compare target values to predicted values and show confusion matrix
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

        # plot confusion matrix
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['Noise', 'P-waves', 'S-waves'])
        disp.plot(cmap='Blues', values_format='')
        plt.title(f'Classification CNN Results ({self.epochs} epochs)')
        plt.tight_layout()
        plt.savefig(self.saved_model_path + '/confusion_matrix.png')
        # plt.show()
        
        # plot accuracy history
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot(self.history.history['accuracy'])
        ax.plot(self.history.history['val_accuracy'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['train','test'])
        plt.savefig(self.saved_model_path + '/model_accuracy.png')
        # plt.show()

    
# Using the class for a classification CNN
s = Seismic(model_type='classification', target='trace_category', choose_dataset_size=-1, full_csv=full_csv, dir=dir) # initialize the class
s.train_test_split(test_size=0.25, random_state=42) # train_test_split
s.classification_cnn(epochs=15) # use the classification cnn method with 15 epochs
s.evaluate_classification_model() # evaluate the model



