#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""


import mne
from tqdm import tqdm
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
    
import pandas as pd
import os
#import spkit as sp
#imfrom scipy.stats import pearsonr, spearmanr

from scipy.signal import butter, sosfilt
from mne_connectivity import spectral_connectivity_epochs,spectral_connectivity_time

def eeg_bandpass(eeg, min_freq, max_freq):
    #this function get signal and range of frequencies we interested and filter out every other frequency components
    # from signal
    fs = 128
    sos_lp = butter(10, max_freq, 'lp', fs=fs, output='sos')
    sos_hp = butter(10, min_freq, 'hp', fs=fs, output='sos')
    
    eeg_low = sosfilt(sos_lp, eeg)
    eeg_high = sosfilt(sos_hp, eeg_low)
    
    return eeg_high

    
def get_label(file, files_data):
    print(file)
    idx = int(file.split("/")[-1].split(".")[0].split("_")[-1])
    print(idx)
    label = np.array(files_data[files_data["participant_id"]  == idx]["Group"])[0]
    if label == "C":
        return [0]
    elif label == "A":
        return [1]
    elif label == "F":
        return [2]



def get_label(file):
    files_data = pd.read_csv('../../../../../ds004504/participants.tsv', sep="\t")
    idx = file.split("/")[-3]
    label = np.array(files_data[files_data["participant_id"] == idx]["Group"])[0]
    if label == "C":
        return [0]
    elif label == "A":
        return [1]
    elif label == "F":
        return [2]
        
    
def get_MMSE(file):
    files_data = pd.read_csv('../../../../../ds004504/participants.tsv', sep="\t")
    idx = file.split("/")[-3]
    label = np.array(files_data[files_data["participant_id"] == idx]["MMSE"])[0]
    return label

    
import json
def data_annotations_caueeg(file):
    f = open('../../../../caueeg/signal/dementia-no-overlap.json')
    data = json.load(f)
    files_metadata = {}
    for file_data in data['train_split']:
        key = file_data['serial']
        value1 = 'train_split'
        value2 = file_data['class_label']
        files_metadata[key] = [value1, [value2]]
    for file_data in data['test_split']:
        key = file_data['serial']
        value1 = 'test_split'
        value2 = file_data['class_label']
        files_metadata[key] = [value1, [value2]]
    for file_data in data['validation_split']:
        key = file_data['serial']
        value1 = 'validation_split'
        value2 = file_data['class_label']
        files_metadata[key] = [value1, [value2]]
    return files_metadata[file.split("/")[-1][:-4]]
    

import json
def data_annotations_caueeg(file):
    f = open('../../../../caueeg/signal/dementia-no-overlap.json')
    data = json.load(f)
    files_metadata = {}
    for file_data in data['train_split']:
        key = file_data['serial']
        value1 = 'train_split'
        value2 = file_data['class_label']
        files_metadata[key] = [value2]
    for file_data in data['test_split']:
        key = file_data['serial']
        value1 = 'test_split'
        value2 = file_data['class_label']
        files_metadata[key] = [value2]
    for file_data in data['validation_split']:
        key = file_data['serial']
        value1 = 'validation_split'
        value2 = file_data['class_label']
        files_metadata[key] = [value2]
    return files_metadata[file.split("/")[-1][:-4]]


def calCorr(eegData, i, j):    
    all_corr = []
    for epoch_idx, epoch in enumerate(eegData):
        corr1 = calculate_cc(epoch, i, j)
        all_corr.append(corr1)
    all_corr = np.array(all_corr)
    return all_corr


def calMI(eegData, i, j):    
    all_mi = []
    for epoch_idx, epoch in enumerate(eegData):
        mi = sp.mutual_info(epoch[i], epoch[j])
        all_mi.append(mi)
    all_mi = np.array(all_mi)
    return all_mi

def calEnt(eegData, i, j):    
    all_en = []
    for epoch_idx, epoch in enumerate(eegData):
        en = sp.entropy_kld(epoch[i], epoch[j])
        all_en.append(en)
    all_en = np.array(all_en)
    return all_en
    
    
def calPLV(eegData, i, j):    
    all_plv = []
    for epoch_idx, epoch in enumerate(eegData):
        plv = hilphase(epoch[i], epoch[j])
        all_plv.append(plv)
    all_plv = np.array(all_plv)
    return all_plv
    
def hilphase(x1,x2):
    sig1_hill=signal.hilbert(x1)
    sig2_hill=signal.hilbert(x2)
    pdt=(np.inner(sig1_hill,np.conj(sig2_hill))/(np.sqrt(np.inner(sig1_hill,
               np.conj(sig1_hill))*np.inner(sig2_hill,np.conj(sig2_hill)))))
    phase = np.angle(pdt)
    return phase
    
# Cross correlation 
# https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/eegraph/strategy.py
def calculate_cc(data_intervals, i, j):
    x = data_intervals[i]
    y = data_intervals[j]
    
    Rxy = signal.correlate(x,y, 'full')
    Rxx = signal.correlate(x,x, 'full')
    Ryy = signal.correlate(y,y, 'full')
    
    lags = np.arange(-len(data_intervals[i]) + 1, len(data_intervals[i]))
    lag_0 = int((np.where(lags==0))[0])

    Rxx_0 = Rxx[lag_0]
    Ryy_0 = Ryy[lag_0]
    
    Rxy_norm = (1/(np.sqrt(Rxx_0*Ryy_0)))* Rxy
    #We use the mean from lag 0 to a 10% displacement. 
    disp = round((len(data_intervals[i])) * 0.10)
    cc_coef = Rxy_norm[lag_0: lag_0 + disp].mean()
    return cc_coef
    

def break_array2chunks(x, epoch_duration):
    size = epoch_duration
    step = int(epoch_duration*0.5)
    trunc = x[x.shape[0]%size]
    x = [x[i : i + size] for i in range(0, len(x), step)]
    return x

def break_all_array(x, size):
    chunk_array = []
    for ch_idx in range(x.shape[1]):
        x_ch = x[:, ch_idx]
        chunck = break_array2chunks(x_ch, size)
        chunk_array.append(chunck[:-2])    
    chunk_array = np.array(chunk_array)
    return chunk_array



import numpy as np

def stack_arrays(x, g, y, task):
    out_x = np.array(x[0])# np.moveaxis(np.float16(x[0]), 0, 1)
    out_g = np.moveaxis(g[0], 0, -1)
    out_g = np.moveaxis(out_g, 0, -1)
    out_y = [y[0] for _ in range(out_x.shape[0])]
    for arr, gr, yl in zip(x[1:], g[1:], y[1:]):
        arr = np.array(arr)
        #arr = np.moveaxis(arr, 0, 1)
        out_x = np.concatenate((out_x, arr), axis=0)
        gr = np.array(gr)
        gr = np.moveaxis(gr, 0, -1)
        gr = np.moveaxis(gr, 0, -1)
        out_g = np.concatenate((out_g, gr), axis=0)
        out_y.extend([yl for _ in range(arr.shape[0])])
    
    out_y_ = []
    if task == "FTD" or task == "FTDandADvsControl":
        for y in out_y:
            if y[0] == 2:
                out_y_.append([y[0]-1])
            else:
                out_y_.append(y)
    elif task == "FTDvsAD":
        for y in out_y:
            out_y_.append([y[0]-1])
    else:
        out_y_ = out_y
    
    out_y = np.array(out_y_)
    #out_x = np.moveaxis(out_x, 1, 2)
    out_x = np.moveaxis(out_x, 2, -1)
    out_g = out_g.reshape(out_x.shape[0], 19, 19, 15)
    return out_x, out_g, out_y
    
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import FastICA, PCA


from scipy import signal
def cal_stft(train_X, test_X, nperseg=256):
    train_stft = []
    test_stft = []
    fs = 256
    WinLength = int(1*fs) # 500 points (0.5 sec, 500 ms)
    step = int(0.05*fs) # 25 points (or 25 ms)
    for x in train_X:
        f, t, Zxx = signal.stft(x, fs, nperseg=nperseg)
        Zxx = np.abs(Zxx)
        Zxx = np.mean(Zxx, -1)
        train_stft.append(Zxx)
    test_stft = []
    for x in test_X:
        f, t, Zxx = signal.stft(x, fs, nperseg=nperseg)
        Zxx = np.abs(Zxx)
        Zxx = np.mean(Zxx, -1)
        test_stft.append(Zxx)

    train_X, test_X = train_stft, test_stft
    
    return train_X, test_X


from scipy import signal
def cal_stft(train_X, test_X, nperseg=128):
    train_stft = []
    test_stft = []
    fs = 256
    
    for x in train_X:
        f, t, Zxx = signal.spectrogram(x, fs, nperseg=nperseg, mode="magnitude", nfft=512)
        Zxx = np.mean(Zxx, -1)
        train_stft.append(Zxx)
    test_stft = []
    for x in test_X:
        f, t, Zxx = signal.spectrogram(x, fs, nperseg=nperseg, mode="magnitude", nfft=512)
        Zxx = np.mean(Zxx, -1)
        test_stft.append(Zxx)

    train_X, test_X = train_stft, test_stft
    
    return train_X, test_X


import numpy as np

def stack_arrays(x, g, y, task):
    out_x = np.array(x[0])# np.moveaxis(np.float16(x[0]), 0, 1)
    out_g = np.moveaxis(g[0], 0, -1)
    out_g = np.moveaxis(out_g, 0, -1)
    out_y = [y[0] for _ in range(out_x.shape[0])]
    for arr, gr, yl in zip(x[1:], g[1:], y[1:]):
        arr = np.array(arr)
        #arr = np.moveaxis(arr, 0, 1)
        out_x = np.concatenate((out_x, arr), axis=0)
        gr = np.array(gr)
        gr = np.moveaxis(gr, 0, -1)
        gr = np.moveaxis(gr, 0, -1)
        out_g = np.concatenate((out_g, gr), axis=0)
        out_y.extend([yl for _ in range(arr.shape[0])])
    
    out_y_ = []
    if task == "FTD" or task == "FTDandADvsControl":
        for y in out_y:
            if y[0] == 2:
                out_y_.append([y[0]-1])
            else:
                out_y_.append(y)
    elif task == "FTDvsAD":
        for y in out_y:
            out_y_.append([y[0]-1])
    else:
        out_y_ = out_y
    
    out_y = np.array(out_y_)
    #out_x = np.moveaxis(out_x, 1, 2)
    out_x = np.moveaxis(out_x, 2, -1)
    out_g = out_g.reshape(out_x.shape[0], 19, 19, 20)
    return out_x, out_g, out_y

import numpy as np

def stack_arrays(x, g, y, task):
    out_g = np.moveaxis(g[0], 0, -1)
    out_g = np.moveaxis(out_g, 0, -1)
    out_y = [y[0] for _ in range(out_g.shape[0])]
    out_x = np.moveaxis(x[0], 0, -1)
    #out_x = np.moveaxis(out_x, 1, 2)
    for arr, gr, yl in zip(x[1:], g[1:], y[1:]):
        gr = np.array(gr)
        gr = np.moveaxis(gr, 0, -1)
        gr = np.moveaxis(gr, 0, -1)
        arr = np.moveaxis(arr, 0, -1)
        #arr = np.moveaxis(arr, 1, 2)
        out_g = np.concatenate((out_g, gr), axis=0)
        out_x = np.concatenate((out_x, arr), axis=0)
        out_y.extend([yl for _ in range(arr.shape[0])])
        
    out_y_ = []
    if task == "FTD" or task == "FTDandADvsControl":
        for y in out_y:
            if y[0] == 2:
                out_y_.append([y[0]-1])
            else:
                out_y_.append(y)
    elif task == "FTDvsAD":
        for y in out_y:
            out_y_.append([y[0]-1])
    else:
        out_y_ = out_y

    out_y = np.array(out_y_)
    out_g = out_g.reshape(out_g.shape[0], 19, 19, 15)
    return out_x, out_g, out_y


import numpy as np

def stack_arrays(x, g, y, task):
    out_g = np.moveaxis(g[0], 0, -1)
    out_g = np.moveaxis(out_g, 0, -1)
    out_y = [y[0] for _ in range(out_g.shape[0])]
    out_x = np.moveaxis(x[0], 1, -1)
    #out_x = np.moveaxis(out_x, 1, 2)
    for arr, gr, yl in zip(x[1:], g[1:], y[1:]):
        gr = np.array(gr)
        gr = np.moveaxis(gr, 0, -1)
        gr = np.moveaxis(gr, 0, -1)
        arr = np.moveaxis(arr, 1, -1)
        #arr = np.moveaxis(arr, 1, 2)
        out_g = np.concatenate((out_g, gr), axis=0)
        out_x = np.concatenate((out_x, arr), axis=0)
        out_y.extend([yl for _ in range(arr.shape[0])])
        
    out_y_ = []
    if task == "FTD" or task == "FTDandADvsControl":
        for y in out_y:
            if y[0] == 2:
                out_y_.append([y[0]-1])
            else:
                out_y_.append(y)
    elif task == "FTDvsAD":
        for y in out_y:
            out_y_.append([y[0]-1])
    else:
        out_y_ = out_y

    out_y = np.array(out_y_)
    out_g = out_g.reshape(out_g.shape[0], 19, 19, 15)
    return out_x, out_g, out_y

import spkit as sp

def normalize(data: np.ndarray, dim=1, norm="l2") -> np.ndarray:
    """Normalizes the data channel by channel

    Args:
        data (np.ndarray): Raw EEG signal
        dim (int, optional): 0 columns, 1 rows. The dimension where 
        the mean and the std would be computed.

    Returns:
        np.ndarray: Channel-wise normalized matrix
    """
    normalized_data = preprocessing.normalize(data, axis=dim, norm=norm)
    return normalized_data

from mne_connectivity import spectral_connectivity_epochs, SpectralConnectivity, envelope_correlation
    


import neurokit2 as nk

from scipy import signal, stats
def cal_features(data):
    fs = 128
    out_dict = {}
    for k, v in data.items():
        signals = v*1e6#+1e-5
        out = np.zeros((signals.shape[0], signals.shape[1], signals.shape[2], 11))
        #signals = signals
        shapes = signals.shape
        for band_idx in tqdm(range(shapes[0])):
            for epoch_idx in range(shapes[1]):
                for ch_idx in range(shapes[2]):
                    signal_ch = signals[band_idx, epoch_idx, ch_idx, :]
                    #signal_ch = (signal_ch - signal_ch.min())/(signal_ch.max() - signal_ch.min())+1e-5
                    #diffen = differential_entropy(signal_ch)
                    diffen, _ = nk.complexity_diffen(signal_ch)
                    if diffen == -np.inf:
                        diffen = 0.0
                    SpEn, _ = nk.entropy_spectral(signal_ch, show=False)
                    complexity, _ = nk.complexity_hjorth(signal_ch)
                    powen, _ = nk.entropy_power(signal_ch)
                    #df, _ = nk.complexity(signal_ch, which = "makowski2022")
                    klen, _ = nk.entropy_kl(signal_ch, delay=1, dimension=3)
                    phasen, _ = nk.entropy_phase(signal_ch, k=4, show=False)
                    #apen, _ = nk.entropy_approximate(signal_ch)
                    #k_max, _ =  nk.complexity_k(signal_ch, k_max='default', show=False)
                    dfd, _ = nk.fractal_density(signal_ch, delay=20, show=False)
                    f, Pxx_den = signal.welch(signal_ch, fs=fs, nperseg=512)
                    Pxx_den = np.sqrt(Pxx_den.mean())
                    kurtosis = stats.kurtosis(signal_ch)
                    mean = np.mean(signal_ch)
                    std = np.std(signal_ch)
                    #sampen, _ = nk.entropy_sample(signal_ch, delay=1, dimension=2)
                    signals_features = [diffen, complexity, SpEn, klen, phasen, powen, 
                                        dfd, Pxx_den, mean, std, kurtosis]
                    out[band_idx, epoch_idx, ch_idx, :] = signals_features
        out_dict[k] = out
        print(out.sum())
    return out_dict

    
def multichannel_sliding_window(X, size, step):
    shape = (X.shape[0] - X.shape[0] + 1, (X.shape[1] - size + 1) // step, X.shape[0], size)
    strides = (X.strides[0], X.strides[1] * step, X.strides[0], X.strides[1])
    return np.lib.stride_tricks.as_strided(X, shape, strides)[0]


def build_data(raw_data, size, files_data, cal_conn=None, raw_eeg=False, 
               bands=True, data_used="dem", idx2label=None, test=False):
    
    eeg_data = []
    
    all_data_features = {}
    data_labels = {}
    data_graphs = {}
    ch_names = {}
    
    for file in tqdm(raw_data):
        #node features
        sample_features = []
        data_features = []
        filtered_eeg = []
        
        if data_used == "caueeg":
            fs = 200
            resamling_fs = 200
            data_raw = mne.io.read_raw_edf(file, verbose=False, preload=True)
    
            ch_names = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F7', 
                        'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Pz', 'Cz']
            ch_types = ['eeg' for _ in range(len(ch_names))]
            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=fs)

            data_raw.info = info
            picks_eeg = mne.pick_types(data_raw.info, meg=False, eeg=True, 
                                       eog=False, stim=False, exclude='bads')
            
            # set montage 10-20
            montage = mne.channels.make_standard_montage("standard_1020")
            data_raw.set_montage(montage)
            data_raw.filter(l_freq=0.5, h_freq=45, verbose=False)
            
        elif data_used == "dem":
            fs = 500
            resamling_fs = 256
            data_raw = mne.io.read_raw(file, verbose=False, preload=True)
            
            montage = mne.channels.make_standard_montage('standard_1020')
            data_raw.set_montage(montage, on_missing='ignore')
            
            data_raw = data_raw.copy().set_montage("standard_1020")
            data_raw.add_reference_channels(["A1", "A2"])
            data_raw.set_eeg_reference(ref_channels=["A1"])
            ch_names = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F7', 
                        'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Pz', 'Cz']
            ch_types = ['eeg' for _ in range(len(ch_names))]
            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=fs)

            data_raw.info = info
            picks_eeg = mne.pick_types(data_raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

        data_raw.filter(l_freq=0.5, h_freq=45, verbose=False)  
                
        data_epochs = data_raw.get_data()
        signal_dur = int(data_epochs.shape[-1]/fs)
        
        data_epochs = signal.resample(data_epochs, signal_dur*resamling_fs, axis=-1)
        data_epochs = multichannel_sliding_window(data_epochs, size*resamling_fs, int(size*resamling_fs/2))
            
        print("epochs", data_epochs.shape)
        
        #get label
        if data_used == "dem":
            label = get_label(file)
        elif data_used == "caueeg":
            try:
                label = data_annotations_caueeg(file)
            except:
                continue
        
        data_labels[file] = label
        
        freq_ranges = [[1, 4],[4, 8], [8, 13], [13, 30], [30, 45]]
        
        
        print("connectivity")
        if cal_conn=="all": 
            conn_methods = ["coh", "pli", "plv"]
            n_con_methods = len(conn_methods)
            n_channels = 19
            n_times = data_epochs.shape[-1]
            n_epochs = data_epochs.shape[0]
            
            n_freq_bands = len(freq_ranges)
            band_c = []
            for freq_band in freq_ranges:
                min_freq = np.min(freq_band)
                max_freq = np.max(freq_band)
                freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 2 + 1))
                nfreqs = len(freqs)
                conns = np.zeros((n_con_methods, n_epochs, n_channels, n_channels, nfreqs))
                conn = spectral_connectivity_time(data_epochs, method=conn_methods, mode="cwt_morlet", 
                                                  sfreq=resamling_fs, freqs=freqs, n_cycles=6,
                                                  verbose=False, fmin=min_freq, fmax=max_freq)
                
                print(conn[0].get_data(output="dense").shape)
                for c in range(n_con_methods):
                    conns[c] = conn[c].get_data(output="dense")
               
                
                conns = np.mean(conns, -1)
                for epoch_conn_idx in range(n_epochs):
                    for conn_idx in range(n_con_methods):
                        conns[conn_idx, epoch_conn_idx, :, :] = np.maximum(conns[conn_idx, epoch_conn_idx, :, :], 
                                                                           conns[conn_idx, epoch_conn_idx, :, :].transpose())
                
                #fil_epochs = eeg_bandpass(data_epochs, min_freq, max_freq)
                #print(fil_epochs.shape)
                #info_conns = np.zeros([1, fil_epochs.shape[0],fil_epochs.shape[1],fil_epochs.shape[1]])
                #
                #for epoch_idx, epoch in enumerate(fil_epochs):
                #    for i in range(19):
                #        for j in range(19):
                #            #mi = sp.mutual_info(fil_epochs[epoch_idx, i, :],fil_epochs[epoch_idx, j, :])
                #            corr = pearsonr(fil_epochs[epoch_idx, i, :], fil_epochs[epoch_idx, j, :]).statistic
                #            info_conns[0, epoch_idx, i, j] = corr
                #            
                #conns = np.concatenate((conns, info_conns), axis=0)
               # 
                band_c.append(conns)

            print("Conns", np.array(band_c).shape)
            data_graphs[file] = band_c
          
          
        signal_dur = int(data_epochs.shape[-1]/resamling_fs)           
        resamling_fs = 128
        data_epochs = signal.resample(data_epochs, signal_dur*resamling_fs, axis=-1)
        """
        filtered_eeg = []
        freq_ranges = [[1, 4], [4, 8], [8, 13], [13, 30], [30, 45]]
        for freq_band in freq_ranges:
            band_epochs = []
            for ch_epoch in data_epochs:
                epoch_signals = []
                for ch_data in ch_epoch:
                    epoch_signal = eeg_bandpass(ch_data, freq_band[0], freq_band[1])
                    epoch_signals.append(epoch_signal)
                band_epochs.append(epoch_signals)
            filtered_eeg.append(band_epochs)
         
        for eeg in filtered_eeg:
            print(np.array(eeg).shape)
        data_features = []
        if raw_eeg:
            for eeg in filtered_eeg:
                data_features.append(signal.resample(eeg, size*resamling_fs, axis=-1)) #sample all signals to same sf
            data_features = np.array(data_features)
        """    
        data_features = data_epochs
        all_data_features[file] = np.array(data_features).squeeze()
        
    return all_data_features, data_graphs, data_labels, ch_names

    
def tr_test_split(test_size):
    path = "mci_ad_dataset_npy/"
    control_subjects_files, AD_subjects_files, MCI_subjects_files = [], [], []
    for file in os.listdir(path):
        if file[:2]=="AD":
            control_subjects_files.append(path+file)
        elif file[:3]=="MCI":
            MCI_subjects_files.append(path+file)
        else:
            AD_subjects_files.append(path+file)

    train_data_files = control_subjects_files + AD_subjects_files + MCI_subjects_files
    train_data_files, test_data_files = train_test_split(train_data_files, test_size=test_size, random_state=100)
    
    return train_data_files, test_data_files