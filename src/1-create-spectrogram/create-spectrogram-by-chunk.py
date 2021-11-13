'''
**This script requires that you have downloaded all data chunks from the STanford EArthquake Dataset (STEAD). The data can be downloaded here: https://github.com/smousavi05/STEAD.
This script reads in the metadata csv files for each data chunk, and allows you to create images from selected waveform signals by pulling the signal data from the hdf5 files. Running the make_images function creates:
        1. Spectrogram plot of signal (ENZ channels correspond to RGB channels)
        
This script runs in parallel using joblib. Set n_jobs to choose number of cores (-1 will use all cores, -2 will use all but one core, etc.)
Adapted from Kaelynn Rose
on 3/11/2021
'''

import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from col_dtypes import ColDataTypes

matplotlib.use('Agg')

############################# INIT VARIABLES #############################

col_dtypes = ColDataTypes()
dtypes = col_dtypes.get_initial_dtype_dict()
date_cols = col_dtypes.get_date_cols()

############################# USER INPUT #############################

data_folder = "../../data/" # root data folder
dpi = 50 # spectrogram image resolution

chunk_id = 2 # chunk 1 is noise, chunks 2-6 are earthquake signals
assert chunk_id > 0 and chunk_id <= 6

data_start = 0 # select start of data rows you want to pull from that chunk
data_end = 200000 # select end of data rows you want to pull from that chunk
data_interval = 500 # select interval you'd like to pull (smaller interval with more loops may run faster)

save_folder = data_folder + 'images/' # folder to save spectrogram images

########################### INIT VARIABLES ############################

csv_pth = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}.csv') # chunk metadata
eqpath = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}.hdf5') # chunk earthquake data

chunk = pd.read_csv(csv_pth, dtype=dtypes, parse_dates=date_cols, encoding='utf-8')
print(eqpath)

Path(save_folder).mkdir(parents=True, exist_ok=True)

#######################################################################

## Make images

eqlist = chunk['trace_name'].to_list()
print(f'No. of waveforms: {len(eqlist)}')
#random_signals = np.random.choice(eqlist,80000,replace=False) # turn on to get random sample of signals
starts = list(np.linspace(data_start, data_end-data_interval, int((data_end-data_start)/data_interval)))
ends = list(np.linspace(data_interval, data_end, int((data_end-data_start)/data_interval)))

# Normalize according to max abs value of ENZ channels (Does not work)
# def normalize_wf(wf_data):
#     abs_max = np.abs(wf_data).max()
#     wf_data /= abs_max
#     wf_data *= 1E8
#     return wf_data

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

def spectrogram_arr(data, dpi):
    # import matplotlib here to prevent multi-processing error
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    fig, ax = plt.subplots(figsize=(3,2), dpi=dpi)
    ax.specgram(data, Fs=100, NFFT=256, cmap='gray', vmin=-10, vmax=25);
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(1)
    ax.set_xlim([0,60])
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.close()

    fig.canvas.draw()
    img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return img_arr

for cnt, n in enumerate(range(len(starts))):
    traces = eqlist[int(starts[n]):int(ends[n])]
    path = eqpath
    
    def make_images(i):
        # retrieving selected waveforms from the hdf5 file:
        try:
            # print('working on waveform ' + str(traces[i]) + ' chunk ' + str(cnt) + ' number ' + str(i))
            dtfl = h5py.File(path, 'r')
            dataset = dtfl.get('data/' + str(traces[i]))

            img_save_pth = save_folder + traces[i] + '.png'
            data = np.array(dataset)
            # data = normalize_wf(data)
            
            # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
            arr_E = spectrogram_arr(data[:,0], dpi)
            # print(arr_E.shape)
            arr_N = spectrogram_arr(data[:,1], dpi)
            # print(arr_N.shape)
            arr_Z = spectrogram_arr(data[:,2], dpi)
            # print(arr_Z.shape)
            
            # img_E = Image.fromarray(arr_E, 'RGB')
            # img_E.save(save_folder + traces[i] + '_E.png')
            # img_N = Image.fromarray(arr_N, 'RGB')
            # img_N.save(save_folder + traces[i] + '_N.png')
            # img_Z = Image.fromarray(arr_Z, 'RGB')
            # img_Z.save(save_folder + traces[i] + '_Z.png')

            channel_ls = [rgb2gray(arr) for arr in [arr_E, arr_N, arr_Z]]
            arr_ENZ = np.stack(channel_ls, axis=2)
            # print("ENZ array:", arr_ENZ.shape)
            
            img = Image.fromarray(arr_ENZ, 'RGB')
            img.save(img_save_pth)    
        except:
            print('String index out of range')
        sys.stdout.write('\rGenerated Spectrograms for batch %04d: %04d of %04d' % (cnt, i+1, len(traces))) # It is okay to have batch_no != len(traces) sometimes due to multi-processing


    # create images for selected data (runs in parallel using joblib)
    start = time.time();
    Parallel(n_jobs=-2)(delayed(make_images)(i) for i in range(0,len(traces))) # run make_images loop in parallel on all but 1 core
    # Parallel(n_jobs=-2)(delayed(make_images)(i) for i in range(0,10))
    # make_images(1000) # to generate a single spectrogram img w/o multi-processing
    end = time.time()
    print(f' | Took {end-start:.5f} s')
