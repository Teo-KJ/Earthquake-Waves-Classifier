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

Created by Kaelynn Rose
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
from skimage import color, filters, io
from skimage.color import rgb2gray
from skimage.transform import resize, rotate
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from model import ClassfierCNN, RegressionCNN

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

            # if choose_dataset_size == 'full':
            #     # select only the rows in the metadata dataframe which correspond to images
            #     print('Selecting traces matching images in directory')
            #     self.img_dataset = self.full_csv.loc[self.full_csv['trace_name'].isin(self.traces_array)] # select rows from the csv that have matching image files
            #     self.labels = self.img_dataset['trace_category'] # target variable, 'earthquake' or 'noise'
            #     self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0)) # transform target variable to numerical categories
            #     print(f'The number of traces in the directory is {len(self.img_dataset)}')
            #     count = 0
            #     for i in range(0,len(self.img_dataset['trace_name'])): # loop through images and read them into the imgs array
            #         count += 0
            #         print(f'Working on trace # {count}')
            #         img = Image.open(self.dir+'/'+self.img_dataset['trace_name'].iloc[i]+'.png') # read in image as grayscale image
            #         self.imgs.append(img)
            #     self.imgs = np.array(self.imgs)
            if type(choose_dataset_size) == int:
                choose_dataset_size = min(min_wave_type_size, choose_dataset_size)
                print(f'Dataset size: {choose_dataset_size}')
                
                # random choice of images from the directory
                choose_p_dataset = random.sample(self.p_traces_array, choose_dataset_size)
                choose_s_dataset = random.sample(self.s_traces_array, choose_dataset_size)
                choose_noise_dataset = random.sample(self.noise_traces_array, choose_dataset_size)

                self.img_dataset = choose_noise_dataset + choose_p_dataset + choose_s_dataset
                self.labels = np.array([0] * choose_dataset_size + [1] * choose_dataset_size + [2] * choose_dataset_size) # target variable, 'noise' is 0 | 'P' is 1 | 'S' is 2
                len_img_dataset = len(self.img_dataset)
                print(f'The number of traces in the directory is {len_img_dataset}')
                for i, img_pth in enumerate(self.img_dataset): # loop through trace names in filtered dataframe and append images to imgs array
                    temp_img = Image.open(img_pth+'.png') # read in image
                    img = temp_img.copy()
                    self.imgs.append(np.asarray(img, dtype=np.uint8))
                    temp_img.close()
                    sys.stdout.write('\rWorking on trace %04d of %04d' % (i+1, len_img_dataset))
                print()
                self.imgs = np.array(self.imgs)   
            else:
                print('Error: please choose either "full" for variable choose_dataset_size to use the full dataset, or provide an integer number of random samples to take from the dataset')
                        
        # elif self.model_type == 'regression':
        #     print(f'The number of all traces in the directory including noise is {len(self.traces_array)}')
        #     local_quakes = self.full_csv[self.full_csv['trace_category'] == 'earthquake_local'] # get only signals corresponding to local earthquakes, not noise
        #     local_quakes_data = local_quakes.loc[local_quakes['trace_name'].isin(self.traces_array)]
            
            # if choose_dataset_size == 'full':
            #     self.img_dataset = local_quakes.loc[local_quakes['trace_name'].isin(self.traces_array)]
            #     print(f'The number of all earthquakes in the directory excluding noise is {len(self.img_dataset)}')
            #     self.labels = self.img_dataset[target] # target variable
                
            #     count = 0
            #     for i in range(0,len(self.img_dataset['trace_name'])): # loop through the images dataframe and read in the images with matching trace names
            #         count += 1
            #         print(f'Working on trace # {count}')
            #         img = Image.open(dir+'/'+self.img_dataset['trace_name'].iloc[i]+'.png') # read in image as grayscale image
            #         self.imgs.append(img)
            #     self.imgs = np.array(self.imgs)
                
            # if type(choose_dataset_size) == int:
            
            #     choose_local_quakes = np.random.choice(np.array(local_quakes_data['trace_name']), choose_dataset_size, replace=False)
            #     self.img_dataset = local_quakes_data.loc[local_quakes_data['trace_name'].isin(choose_local_quakes)]
            #     self.labels = self.img_dataset[self.target] # target variable
                
            #     count = 0
            #     for i in range(0,len(self.img_dataset['trace_name'])): # loop through dataframe and read in images corresponding to trace names in data frame
            #         count += 1
            #         print(f'Working on trace # {count}')
            #         img = Image.open(dir+'/'+self.img_dataset['trace_name'].iloc[i]+'.png') # read in image as grayscale image
            #         self.imgs.append(img)
            #     self.imgs = np.array(self.imgs)
                
            # else:
            #     print('Error: please choose either "full" for variable choose_dataset_size to use the full dataset, or provide an integer number of random samples to take from the dataset')
                
        else:
            print('Error: please choose either "classification" or "regression" for CNN model type')
 

    def train_test_split(self, test_size, random_state):
        # train test split using sklearn
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self.imgs, self.labels, random_state=random_state, test_size=test_size)
        print(f'Training images shape:\t{self.train_images.shape}')
        print(f'Training labels shape:\t{self.train_labels.shape}')
        print(f'Testing images shape:\t{self.test_images.shape}')
        print(f'Testing labels shape:\t{self.test_labels.shape}')
        
        print('Scaling image intensity')
        self.train_images = self.train_images/255.0 # scale intensity to between 0 and 1
        self.test_images = self.test_images/255.0 # scale intensity to between 0 and 1

        img_height = self.train_images.shape[1] # get height of each image in pixels
        img_width = self.train_images.shape[2] # get width of each image in pixels

        print('Resizing images')
        self.train_images = self.train_images.reshape(-1,img_height,img_width,3) # reshape to input into CNN which requires a 4-tensor
        self.test_images = self.test_images.reshape(-1,img_height,img_width,3) # reshape to input into CNN which requires a 4-tensor

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
        img_shape = self.train_images.shape[1:]
        model = ClassfierCNN.create_model(img_shape)

        self.history = model.fit(self.train_images, self.train_labels, epochs=epochs, callbacks=callbacks, validation_split=0.2) # fit model and save history

        print(model.summary())
        
        # Set model save path
        saved_model_path = f'./saved_models/waves_{str(self.choose_dataset_size)}dataset_{self.model_type}_{self.target}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}'
        # Save entire model to a HDF5 file
        model.save(saved_model_path)
        self.model = model
        
    # def regression_cnn(self, epochs):
    #     self.epochs = epochs
        
    #     # set callbacks so that the model will be saved after each epoch
    #     callbacks = [
    #         keras.callbacks.ModelCheckpoint(
    #             filepath=f'./saved_models/waves_{str(self.choose_dataset_size)}dataset_{self.model_type}_{self.target}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}',
    #             save_freq='epoch')
    #     ]

    #     # build CNN on dataset
    #     print('Building Regression CNN model')
    #     model = RegressionCNN.create_model()

    #     self.history = model.fit(self.train_images, self.train_labels, epochs=epochs, callbacks=callbacks, validation_split=0.2) # fit model and save model loss history

    #     print(model.summary())
        
    #     # Set model save path
    #     saved_model_path = f'./saved_models/waves_{str(self.choose_dataset_size)}dataset_{self.model_type}_{self.target}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}' # _%H%M%S
    #     # Save entire model to a HDF5 file
    #     model.save(saved_model_path)
    #     self.model = model


    def evaluate_classification_model(self):
        print('Evaluating model on test dataset')
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=0) # get model evaluation metrics
        print("\nTest data, accuracy: {:5.2f}%".format(100*self.test_acc))

        print('Finding predicted classes and probabilities to build confusion matrix')
        self.predicted_classes = np.argmax(self.model.predict(self.test_images),axis=-1) # predict the class of each image
        self.predicted_probs = self.model.predict(self.test_images) # predict the probability of each image belonging to a class

        # create confusion matrix
        print('Confusion matrix:')
        self.cm = confusion_matrix(self.test_labels,self.predicted_classes) # compare target values to predicted values and show confusion matrix
        print(self.cm)

        accuracy = accuracy_score(self.test_labels,  self.predicted_classes)
        print(f'Accuracy: {accuracy}')

        # Micro average metrics: 
        # Calculate metrics globally by counting the total TPs, FNs & FPs.
        precision = precision_score(self.test_labels, self.predicted_classes, average='micro')
        recall = recall_score(self.test_labels, self.predicted_classes, average='micro')
        f1 = f1_score(self.test_labels, self.predicted_classes, average='micro')
        print('[Micro Average]:')
        print(f'Precision: {precision}\nRecall: {recall}\nF1 score: {f1}')

        # Weighted average metrics: 
        # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label) to account for label imbalance.
        # NOTE: can result in an F-score that is not between precision and recall
        precision = precision_score(self.test_labels, self.predicted_classes, average='weighted')
        recall = recall_score(self.test_labels, self.predicted_classes, average='weighted')
        f1 = f1_score(self.test_labels, self.predicted_classes, average='weighted', labels=np.unique(self.predicted_classes))
        print('[weighted Average]:')
        print(f'Precision: {precision}\nRecall: {recall}\nF1 score: {f1}')

        # plot confusion matrix
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['Noise', 'P-waves', 'S-waves'])
        disp.plot(cmap='Blues', values_format='')
        plt.title(f'Classification CNN Results ({self.epochs} epochs)')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
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
        plt.savefig('model_accuracy.png')
        # plt.show()
        
    def evaluate_regression_model(self):
        print('Evaluating model on test dataset')
        self.test_loss = self.model.evaluate(self.test_images, self.test_labels, verbose=1) # get model evaluation metrics
        print(f'Test data loss: {self.test_loss}')
        
        print('Getting predictions')
        self.predicted = self.model.predict(self.test_images) # get target value predictions for each image

        # plot scatterplot of observed values vs. predicted values for target
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(7,7))
        ax.scatter(self.test_labels,self.predicted,alpha=0.25)
        point1 = [0,0]
        point2 = [6,6]
        xvalues = [point1[0], point2[0]]
        yvalues = [point1[1], point2[1]]
        ax.plot(xvalues,yvalues,color='blue')
        ax.set_ylabel('Predicted Value')
        ax.set_xlabel('Observed Value')
        ax.set_title(f'Regression CNN Results ({self.epochs} epochs)')
        plt.tight_layout()
        plt.savefig('true_vs_predicted.png')
        plt.show()

        # plot model loss history
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot(self.history.history['loss'])
        ax.plot(self.history.history['val_loss'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['train','test'])
        plt.savefig('model_loss.png')
        plt.show()
        

    
# Using the class for a classification CNN
s = Seismic(model_type='classification', target='trace_category', choose_dataset_size=50000, full_csv=full_csv, dir=dir) # initialize the class
s.train_test_split(test_size=0.25, random_state=42) # train_test_split
s.classification_cnn(epochs=15) # use the classification cnn method with 15 epochs
s.evaluate_classification_model() # evaluate the model

# Using the class for a regression CNN
# s = Seismic(model_type='regression',target='source_magnitude',choose_dataset_size=60000,full_csv,dir) # initialize the class
# s.train_test_split(test_size=0.25,random_state=41) # train_test_split
# s.regression_cnn(target='source_magnitude',epochs=15) # use the regression cnn method with 15 epochs with a target variable
# s.evaluate_regression_model() # evaluate the model



