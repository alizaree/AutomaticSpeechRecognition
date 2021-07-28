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
from itertools import permutations

def PIT_MSE(estimation, target):
    """
    args:
        estimation: Pytorch tensor, shape (batch_size, num_target, freq, time)
        target: Pytorch tensor, shape (batch_size, num_target, freq, time)
    output:
        mse: single scalar, average MSE across all dimensions
    """
    
    batch_size, C, freq, time = estimation.shape
    
    all_perm = list(permutations(range(C)))
    all_mse = []
    for i in range(len(all_perm)):
        this_mse = (estimation - target[:,all_perm[i]]).pow(2).sum(2).view(batch_size, -1).mean(1)
        all_mse.append(this_mse.unsqueeze(1))
        
    all_mse = torch.cat(all_mse, 1)  # batch_size, num_perm
    mse, _ = torch.min(all_mse, 1)
    
    return mse.mean()

class DeepLSTM(nn.Module):
    def __init__(self, feature_dim, num_spk):
        super(DeepLSTM, self).__init__()
        
        # layers, and activation function
        
        self.feature_dim = feature_dim
        self.hidden_unit = 128  # number of hidden units
        self.num_spk = num_spk
        
        # TODO: first LSTM layer
        self.LSTM1 = nn.LSTM(self.feature_dim, self.hidden_unit, num_layers=1, batch_first=True)
        
        # TODO: second LSTM layer
        self.LSTM2 = nn.LSTM(self.hidden_unit, self.hidden_unit, num_layers=1, batch_first=True)
        
        # output layer
        self.output = nn.Linear(self.hidden_unit, self.feature_dim*self.num_spk)
        
    # the function for the forward pass of network (i.e. from input to output)
    def forward(self, input):
        # the input is a batch of spectrograms with shape (batch_size, frequency_dim, time_step)
        
        batch_size, freq_dim, time_step = input.shape
        
        # MVN
        batch_mean = input.mean(2)  # (batch, freq)
        batch_std = (input.var(2) + 1e-6).sqrt()  # (batch, freq)
        MVN_input = (input - batch_mean.unsqueeze(2)) / batch_std.unsqueeze(2)  #  (batch, freq, time)
        
        # reshape the input
        
        MVN_input = MVN_input.transpose(1, 2).contiguous()  # (batch, time, freq), swap the two dimensions
        
        # pass it through the 2 LSTM layers
        
        output, output_hidden = self.LSTM1(MVN_input, None)
        output, output_hidden = self.LSTM2(output, None)
        
        # the output should have shape (batch, time, hidden_unit)
        # pass to the output layer
        
        mask = self.output(output.contiguous().view(batch_size*time_step, -1))  # *batch*time, freq*num_spk
        
        # reshape back
        mask = mask.view(batch_size, time_step, freq_dim*self.num_spk)  # (batch, time, freq*num_spk)
        mask = mask.transpose(1, 2).contiguous().view(batch_size, self.num_spk, freq_dim, time_step)  # (batch, num_spk, freq, time)
        # Softmax activation
        mask = torch.softmax(mask, dim=1)
        
        return mask
    
def train(train_loader, model, optimizer ,epoch,log_step, versatile=True):

    start_time = time.time()
    model = model.train()  # set the model to training mode. Always do this before you start training!
    train_loss = 0.
    
    # load batch data
    for batch_idx, data in enumerate(train_loader):
        mixture_spec, spk1_spec, spk2_spec = data
        
        # clean up the gradients in the optimizer
        # this should be called for each batch
        optimizer.zero_grad()
        
        mask = model(mixture_spec)
        
        # calculate oracle WFM
        WFM1 = spk1_spec.pow(2) / (spk1_spec.pow(2) + spk2_spec.pow(2) + 1e-6)
        WFM2 = spk2_spec.pow(2) / (spk1_spec.pow(2) + spk2_spec.pow(2) + 1e-6)
        
        WFM = torch.cat([WFM1.unsqueeze(1), WFM2.unsqueeze(1)], 1)  # batch_size, 2, freq, time
        
        # MSE as objective
        loss = PIT_MSE(mask * mixture_spec.unsqueeze(1), WFM * mixture_spec.unsqueeze(1))
        
        # automatically calculate the backward pass
        loss.backward()
        # perform the actual backpropagation
        optimizer.step()
        
        train_loss += loss.data.item()
        
        # OPTIONAL: you can print the training progress 
        if versatile:
            if (batch_idx+1) % log_step == 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | MSE {:5.4f} |'.format(
                    epoch, batch_idx+1, len(train_loader),
                    elapsed * 1000 / (batch_idx+1), 
                    train_loss / (batch_idx+1)
                    ))
    
    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | MSE {:5.4f} |'.format(
            epoch, (time.time() - start_time), train_loss))
    
    return train_loss
        
def validate(model, epoch):
    start_time = time.time()
    model = model.eval()  # set the model to evaluation mode. Always do this during validation or test phase!
    validation_loss = 0.
    
    # load batch data
    for batch_idx, data in enumerate(validation_loader):
        mixture_spec, spk1_spec, spk2_spec = data
        
        # you don't need to calculate the backward pass and the gradients during validation
        # so you can call torch.no_grad() to only calculate the forward pass to save time and memory
        with torch.no_grad():
            
            # calculate oracle WFM
            WFM1 = spk1_spec.pow(2) / (spk1_spec.pow(2) + spk2_spec.pow(2) + 1e-6)
            WFM2 = spk2_spec.pow(2) / (spk1_spec.pow(2) + spk2_spec.pow(2) + 1e-6)

            WFM = torch.cat([WFM1.unsqueeze(1), WFM2.unsqueeze(1)], 1)  # batch_size, 2, freq, time
        
            mask = model(mixture_spec)
        
            # MSE as objective
            loss = PIT_MSE(mask * mixture_spec.unsqueeze(1), WFM * mixture_spec.unsqueeze(1))
        
            validation_loss += loss.data.item()
    
    validation_loss /= (batch_idx+1)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | MSE {:5.4f} |'.format(
            epoch, (time.time() - start_time), validation_loss))
    print('-' * 99)
    
    return validation_loss