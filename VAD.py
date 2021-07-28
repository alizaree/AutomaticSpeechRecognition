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

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# TODO: use the LinearSVC class for the VAD data
# Set random_state to 0 and max_iter to 1e3 for a fair comparison
def SVM_VAD(training_data, training_label):
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, max_iter=1e5))
    clf.fit(training_data.T, training_label)

    # TODO: make prediction on the training, validation, and test sets
    # Also print the classification accuracy
    trn_pred=clf.predict(training_data.T)
    trn_acs=clf.score(training_data.T, training_label)


    print('mean accuracy of training prediction:', trn_acs)
    return clf, trn_pred, trn_acs


def smooth_pred(tst_pred,test_label, w_L=3 ):
    # TODO: smoothing for Linear-SVM output label, print the accuracy
    len_T=len(tst_pred)
    smoothed_svm=[idx if idd <w_L or idd>len_T-(w_L+1) else (np.mean(tst_pred[idd-w_L:idd+(w_L+1)])>0.5)*1 for idd, idx in enumerate(tst_pred) ]

    #Calculating the accs

    ACC_nonsmth_svm=np.mean(tst_pred==test_label)*100
    
    ACC_smth_svm=np.mean(smoothed_svm==np.array(test_label))*100
    
    print('Acc smoothed svm:',ACC_smth_svm)
    print('Acc non smoothed svm:',ACC_nonsmth_svm)
    
    return smoothed_svm, ACC_smth_svm

def seg_out(Trn_pred, training_data ):
    pred=np.array(Trn_pred)
    pred1=np.concatenate((np.array([0]),pred ), axis=0)
    pred2=np.concatenate((pred,np.array([0]) ), axis=0)
    Starts=(pred2!=pred1) & pred2
    Ends=(pred2!=pred1) & pred1
    id_str=[idd for idd , item in enumerate(Starts[:-1]) if item==1]
    id_end=[idd for idd , item in enumerate(Ends[:-1]) if item==1]
    out=[ training_data[:, id_str[idd]:id_end[idd] ] for idd in range(len(id_str))]
    return out, id_str, id_end

def Time_finder(label):
    pred=np.array(label)
    pred1=np.concatenate((np.array([0]),pred ), axis=0)
    pred2=np.concatenate((pred,np.array([0]) ), axis=0)
    Starts=(pred2!=pred1) & pred2
    Ends=(pred2!=pred1) & pred1
    id_str=[idd for idd , item in enumerate(Starts[:-1]) if item==1]
    id_end=[idd for idd , item in enumerate(Ends[:-1]) if item==1]
    return id_str, id_end
    
