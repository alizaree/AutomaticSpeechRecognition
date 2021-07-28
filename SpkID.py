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
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


class AlexNet(nn.Module):
    def __init__(self, num_classes=50):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=2),  # number of input channel is 1 (for image it is 3) 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # we make the number of hidden channels smaller in these layers
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))  # perform adaptive mean pooling on any size of the input to match the provided size
        self.classifier = nn.Sequential(
            # nn.Dropout()  no Droupout layers here
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout()  no Droupout layers here
            nn.Linear(256, 256),
            nn.Tanh() # use Tanh instead of ReLU since the output here will be used for the speaker embeddings
        )
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # the dimension after adaptive average pooling is (batch, 64, 3, 3)
        x = torch.flatten(x, 1)  # average
        x = self.classifier(x)  
        # normalize before the last layer
        embedding = x / (x.pow(2).sum(1) + 1e-6).sqrt().unsqueeze(1)
        # output layer
        x = self.output(embedding)
        return x, embedding
    
class dataset_pipeline(Dataset):
    def __init__(self, data, label):
        super(dataset_pipeline, self).__init__()
        
        self.data = data
        self.label = label
        
        self._len = data.shape[0]  # number of utterances
    
    def __getitem__(self, index):
        # calculate STFT here
        spec=torch.from_numpy(np.squeeze(self.data[index,:,:]).astype(np.float32))  
        label = self.label[index]
        label = torch.from_numpy(np.array(label)).long()
        return spec, label
    
    def __len__(self):
        return self._len
# CE loss
def CE(output, target):
    # output shape: (batch, num_classes)
    # target shape: (batch,)
    
    loss = nn.CrossEntropyLoss()
    
    return loss(output, target)


def train2(train_loader, model, optimizer ,epoch,log_step, versatile=True):
    start_time = time.time()
    model = model.train()  # set the model to training mode. Always do this before you start training!
    train_loss = 0.
    correct = 0
    total = 0
    # load batch data
    for batch_idx, data in enumerate(train_loader):
        spec, label = data
        
        optimizer.zero_grad()
        
        output, output_embedding = model(spec.unsqueeze(1))
        
        # CE as objective
        loss = CE(output, label)
        
        # automatically calculate the backward pass
        loss.backward()
        # perform the actual backpropagation
        optimizer.step()
        
        train_loss += loss.data.item()
        
        _, output_label = torch.max(output, 1)
        output_label = output_label.data.numpy()
        label = label.data.numpy()
        correct += np.sum(output_label == label)
        total += len(label)
        
        
        # OPTIONAL: you can print the training progress 
        if versatile:
            if (batch_idx+1) % log_step == 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | CE {:5.4f} |'.format(
                    epoch, batch_idx+1, len(train_loader),
                    elapsed * 1000 / (batch_idx+1), 
                    train_loss / (batch_idx+1)
                    ))
    
    train_loss /= (batch_idx+1)
    accuracy = correct / total
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | CE {:5.4f} | Accuracy {:5.4f} |'.format(
            epoch, (time.time() - start_time), train_loss, accuracy))
    
    return train_loss
def validate(model, epoch):
    start_time = time.time()
    model = model.eval()  # set the model to evaluation mode. Always do this during validation or test phase!
    correct = 0
    total = 0
    
    # load batch data
    for batch_idx, data in enumerate(validation_loader):
        spec, label = data
        
        # you don't need to calculate the backward pass and the gradients during validation
        # so you can call torch.no_grad() to only calculate the forward pass to save time and memory
        with torch.no_grad():
        
            output, output_embedding = model(spec.unsqueeze(1))
        
            # calculate accuracy
            _, output_label = torch.max(output, 1)
            output_label = output_label.data.numpy()
            label = label.data.numpy()
            correct += np.sum(output_label == label)
            total += len(label)
        
    accuracy = correct / total
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | Accuracy {:5.4f} |'.format(
            epoch, (time.time() - start_time), accuracy))
    print('-' * 99)
    
    return accuracy

def EER(y, y_pred):
    # y_pred is a list of similarity scores for the verification task (cosine similarity values)
    # y is a list of the binary labels (accept/reject), where 1 is used for acceptance
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    decision_threshold = interp1d(fpr, threshold)(eer)
    
    return eer, decision_threshold


def sample_triplet( training_data, training_label, mode='train'):
    nSpk=max(training_label)+1
    '''
    if mode == 'train':
        #sample_num = 10
        #dataset = train_audio
    elif mode == 'val':
        #sample_num = 3
        #dataset = val_audio
    '''  
    # sample two speaker indices
    spk_idx = np.random.choice(nSpk, 2, replace=False)
    
    # for the first speaker, sample two utterance indices
    # each speaker has 10 utterance for training
    # first speaker's utterances:
    dataSpk1=training_data[ [idd for idd in training_label if idd==spk_idx[0] ], :,:]
    anchor_idx, positive_idx = np.random.choice(dataSpk1.shape[0], 2, replace=False)
        
    # for the second speaker, sample one utterance index
    dataSpk2=training_data[ [idd for idd in training_label if idd==spk_idx[1] ], :,:]
    negative_idx = np.random.choice(dataSpk2.shape[0])
        
    # load the utterances
    anchor_utterance = torch.from_numpy(dataSpk1[anchor_idx,:,:].astype(np.float32))
    positive_utterance = torch.from_numpy(dataSpk1[positive_idx,:,:].astype(np.float32))
    negative_utterance = torch.from_numpy(dataSpk2[negative_idx,:,:].astype(np.float32))
        
    
    return anchor_utterance, positive_utterance, negative_utterance


# triplet loss
def triplet_loss(theta_ap, theta_an, alpha=1):
    # theta_ap shape: (batch,)
    # theta_an shape: (batch,)
    
    loss = F.relu(theta_an - theta_ap + alpha)
    
    return loss.mean()
def train(train_loader,training_data, training_label, model, optimizer ,epoch,log_step, versatile=True):
    start_time = time.time()
    model = model.train()  # set the model to training mode. Always do this before you start training!
    train_loss = 0.
    
    # load batch data
    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        
        # triplet loss
        batch_anchor_spec = []
        batch_positive_spec = []
        batch_negative_spec = []
        batch_size=data[0].shape[0]
        for j in range(batch_size):
            anchor_spec, positive_spec, negative_spec = sample_triplet( training_data, training_label, mode='train')
            batch_anchor_spec.append(anchor_spec.unsqueeze(0))
            batch_positive_spec.append(positive_spec.unsqueeze(0))
            batch_negative_spec.append(negative_spec.unsqueeze(0))
            
        batch_anchor_spec = torch.cat(batch_anchor_spec, 0)
        batch_positive_spec = torch.cat(batch_positive_spec, 0)
        batch_negative_spec = torch.cat(batch_negative_spec, 0)
        
        # get the embeddings
        _ , anchor_embedding = model(batch_anchor_spec.unsqueeze(1))
        _ , positive_embedding = model(batch_positive_spec.unsqueeze(1))
        _ , negative_embedding = model(batch_negative_spec.unsqueeze(1))
        
        # calculate cosine similarity scores
        theta_ap = (anchor_embedding * positive_embedding).sum(1)
        theta_an = (anchor_embedding * negative_embedding).sum(1)
        
        # triplet loss
        loss = triplet_loss(theta_ap, theta_an)
        
        # automatically calculate the backward pass
        loss.backward()
        # perform the actual backpropagation
        optimizer.step()
        
        train_loss += loss.data.item()
        
        # OPTIONAL: you can print the training progress 
        if versatile:
            if (batch_idx+1) % log_step == 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | triplet loss {:5.4f} |'.format(
                    epoch, batch_idx+1, len(train_loader),
                    elapsed * 1000 / (batch_idx+1), 
                    train_loss / (batch_idx+1)
                    ))
    
    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | triplet loss {:5.4f} |'.format(
            epoch, (time.time() - start_time), train_loss))
    
    return train_loss