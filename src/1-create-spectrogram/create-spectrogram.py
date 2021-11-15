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

########################## INIT CSV DTYPES ###########################

col_dtypes = ColDataTypes()
dtypes = col_dtypes.get_initial_dtype_dict()
date_cols = col_dtypes.get_date_cols()

############################# USER INPUT #############################

data_folder = "../../data/" # root data folder
min_acceptable_duration = 500 # min fixed window duration

# chunks to generate spectrogram imgs from
chunk_ids = [1, 2] # chunk 1 is noise, chunks 2-6 are earthquake signals
assert min(chunk_ids) > 0 and max(chunk_ids) <= 6

data_start = 0 # select start of data rows you want to pull from that chunk
data_end = 200000 # select end of data rows you want to pull from that chunk
data_interval = 500 # select interval you'd like to pull (smaller interval with more loops may run faster)

# multi-processing
n_jobs = -2 # run on all cores except 1

# plotting
dpi = 50 # image resolution
figsize = (3,2)

# spectrogram related params
sampling_freq = 100
NFFT = 50
noverlap = 25

# save folders
save_folder = data_folder + 'images/' # folder to save spectrogram images
noise_folder = save_folder + 'noise/' # folder to save noise spectrogram images
p_folder = save_folder + 'P/' # folder to save P-wave spectrogram images
s_folder = save_folder + 'S/' # folder to save S-wave spectrogram images

############################### INIT ################################

def get_paths(chunk_id, processed=True):
    if processed:
        csv_pth = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}_processed.csv') # chunk metadata
    else:
        csv_pth = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}.csv') # chunk metadata

    eqpath = os.path.join(data_folder, f'raw/chunk{chunk_id}/chunk{chunk_id}.hdf5') # chunk earthquake data
    return csv_pth, eqpath

data_dict = {}
for chunk_id in chunk_ids:
    processed = False if chunk_id == 1 else True
    csv_pth, eqpath = get_paths(chunk_id, processed=processed)
    data_dict[chunk_id] = {
        'csv': csv_pth,
        'eqpath': eqpath,
    }

xlim = min_acceptable_duration / sampling_freq

Path(save_folder).mkdir(parents=True, exist_ok=True)
Path(noise_folder).mkdir(parents=True, exist_ok=True)
Path(p_folder).mkdir(parents=True, exist_ok=True)
Path(s_folder).mkdir(parents=True, exist_ok=True)

############################# FUNCTIONS #############################

## Spectrogram related fns
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

def spectrogram_arr(data, dpi):
    # import matplotlib here to prevent multi-processing error
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.specgram(data, Fs=sampling_freq, NFFT=NFFT, noverlap=noverlap, cmap='gray', vmin=-10, vmax=25);
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(1)
    ax.set_xlim([0, xlim])
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

def get_ENZ_spectrogram_arr(data, dpi):
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    arr_E = spectrogram_arr(data[:,0], dpi)
    arr_N = spectrogram_arr(data[:,1], dpi)
    arr_Z = spectrogram_arr(data[:,2], dpi)
    
    # img_E = Image.fromarray(arr_E, 'RGB')
    # img_E.save(save_folder + traces[i] + '_E.png')
    # img_N = Image.fromarray(arr_N, 'RGB')
    # img_N.save(save_folder + traces[i] + '_N.png')
    # img_Z = Image.fromarray(arr_Z, 'RGB')
    # img_Z.save(save_folder + traces[i] + '_Z.png')

    channel_ls = [rgb2gray(arr) for arr in [arr_E, arr_N, arr_Z]]
    arr_ENZ = np.stack(channel_ls, axis=2)
    
    return arr_ENZ

def get_wave_arrivals(trace_info, min_acceptable_duration):
    p_arrival = int(trace_info['p_arrival_sample'])
    s_arrival = int(trace_info['s_arrival_sample'])
    # print(p_arrival, s_arrival)
    
    return {
        'P': slice(p_arrival, p_arrival + min_acceptable_duration),
        'S': slice(s_arrival, s_arrival + min_acceptable_duration)
    }

def make_eq_images(i, traces_csv):
    # retrieving selected waveforms from the hdf5 file:
    try:
        dtfl = h5py.File(path, 'r')
        trace_name = str(traces[i])
        dataset = dtfl.get('data/' + trace_name)

        data = np.array(dataset)
        
        trace_info = traces_csv.loc[traces_csv['trace_name'] == trace_name, :]
        wave_arrivals = get_wave_arrivals(trace_info, min_acceptable_duration)

        arr_P_ENZ = get_ENZ_spectrogram_arr(data[wave_arrivals['P'], :], dpi)
        arr_S_ENZ = get_ENZ_spectrogram_arr(data[wave_arrivals['S'], :], dpi)
        # print("ENZ P/S array:", arr_P_ENZ.shape, arr_S_ENZ.shape)

        img_P = Image.fromarray(arr_P_ENZ, 'RGB')
        img_S = Image.fromarray(arr_S_ENZ, 'RGB')

        p_save_pth = p_folder + trace_name + '.png'
        s_save_pth = s_folder + trace_name + '.png'
        img_P.save(p_save_pth)
        img_S.save(s_save_pth)    
    except Exception as e:
        print(e)
    sys.stdout.write('\rGenerated Spectrograms for batch %04d: %04d of %04d' % (cnt, i+1, len(traces))) # It is okay to have batch_no != len(traces) sometimes due to multi-processing

def make_noise_images(i):
    # retrieving selected waveforms from the hdf5 file:
    try:
        dtfl = h5py.File(path, 'r')
        dataset = dtfl.get('data/' + str(traces[i]))

        img_save_pth = noise_folder + traces[i] + '.png'
        data = np.array(dataset)
        
        slice_start = np.random.randint(data.shape[0] - min_acceptable_duration)
        noise_slice = slice(slice_start, slice_start + min_acceptable_duration)
        
        arr_ENZ = get_ENZ_spectrogram_arr(data[noise_slice, :], dpi)
        # print("ENZ noise array:", arr_ENZ.shape)

        img = Image.fromarray(arr_ENZ, 'RGB')
        img.save(img_save_pth)    
    except Exception as e:
        print(e)
    sys.stdout.write('\rGenerated Spectrograms for batch %04d: %04d of %04d' % (cnt, i+1, len(traces))) # It is okay to have batch_no != len(traces) sometimes due to multi-processing


############################### MAIN ################################

## Make images
for id, id_info_dict in data_dict.items():
    chunk_csv = pd.read_csv(csv_pth, dtype=dtypes, parse_dates=date_cols, encoding='utf-8')
    eqpath = id_info_dict['eqpath']
    eqlist = chunk_csv['trace_name'].to_list()
    print(f'No. of waveforms in chunk {id}: {len(eqlist)}')

    data_end = min(len(eqlist), data_end)
    #random_signals = np.random.choice(eqlist,80000,replace=False) # turn on to get random sample of signals
    starts = list(np.linspace(data_start, data_end-data_interval, int((data_end-data_start)/data_interval)))
    ends = list(np.linspace(data_interval, data_end, int((data_end-data_start)/data_interval)))

    for cnt, n in enumerate(range(len(starts))):
        traces = eqlist[int(starts[n]):int(ends[n])]
        traces_csv = chunk_csv.loc[int(starts[n]):int(ends[n]), ['trace_name', 'p_arrival_sample', 's_arrival_sample']]
        path = eqpath

        # create images for selected data (runs in parallel using joblib)
        start = time.time()
        if id == 1:
            Parallel(n_jobs=n_jobs)(delayed(make_noise_images)(i) for i in range(0, len(traces)))
        else:
            Parallel(n_jobs=n_jobs)(delayed(make_eq_images)(i, traces_csv) for i in range(0, len(traces)))
        end = time.time()
        print(f' | Took {end-start:.5f} s')
