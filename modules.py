import numpy as np
import librosa
import os
import time
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from IPython.display import Audio

from torch.utils.data import Dataset, DataLoader

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

def get_filters(filter_points, FFT_size):
    
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters



def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis


def Create_Mel( audio, white_noise=True, dct_filter_num=26, npc=13 ,sr=16000,hop_length=256, Plot=True):
    audio_spec = librosa.stft(audio, n_fft=512, hop_length=hop_length)
    spec_phase = np.angle(audio_spec)
    audio_spec_mag = np.abs(audio_spec)
    print('shape of spec ', audio_spec_mag.shape)
    # TODO: calculate MFCC using the magnitude spectrogram above
    # use 64 filters for Mel filterbank
    # name your output as mel_spec

    ## reference: https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    freq_min = 0
    freq_high = sr / 2
    mel_filter_num=64
    FFT_size=512
    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=sr)
    filters = get_filters(filter_points, FFT_size)
    # taken from the librosa library This is being done to prevent the noise increase because if the increase of bandwidth
    if white_noise:
        enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
        filters *= enorm[:, np.newaxis]
        

    if Plot:
        plt.figure(figsize=(15,4))
        for n in range(filters.shape[0]):
            plt.plot(filters[n])

    print('filter shape', filters.shape)   
    audio_filtered = np.dot(filters, (audio_spec_mag)**2)+1e-8
    print('minV',np.amin(audio_filtered))
    mel_logP = 10.0 * np.log10(audio_filtered)
    
    #dct_filter_num = 26 # reduce the dimenssion from  filterbank=64 to 26 DCTs and then only keep 12 lower freqs

    dct_filters = dct(dct_filter_num, mel_filter_num)

    cepstral_coefficents = np.dot(dct_filters, mel_logP)
    mel_spec=cepstral_coefficents[:npc, :]
    print('min',np.amin(mel_spec))



    print('shape of mel:' , mel_spec.shape)  # freq, time_steps
    
            
    return mel_spec, audio_spec_mag, spec_phase


def frame_label( label, hop_length=256):
    max_L=len(label)//hop_length+1
    new_label=np.zeros(max_L)
    for sample in range(len(label)):
        new_label[sample//hop_length]+=label[sample]
    new_label=[1 if idd>0 else 0 for idd in new_label]
    return new_label


def make_batch(a, batch_size):
        b=a.reshape(a.shape[0],batch_size,a.shape[1]//batch_size).transpose(1,0,2)
        return b
        


# a class to load the saved h5py dataset
class dataset_pipeline(Dataset):
    def __init__(self, mixture, speech1, speech2):
        super(dataset_pipeline, self).__init__()

        #self.h5pyLoader = h5py.File(path, 'r')
        
        self.mixture = mixture
        self.speech1 = speech1
        self.speech2 = speech2
        
        self._len = self.mixture.shape[0]  # number of utterances
    
    def __getitem__(self, index):
        # calculate STFT here
        mixture_spec = torch.from_numpy(np.squeeze(self.mixture[index,:,:]).astype(np.float32)) # only use the magnitude spectrogram
        speech1_spec = torch.from_numpy(np.squeeze(self.speech1[index,:,:]).astype(np.float32))
        speech2_spec = torch.from_numpy(np.squeeze(self.speech2[index,:,:]).astype(np.float32))
            
        return mixture_spec, speech1_spec, speech2_spec
    
    def __len__(self):
        return self._len


