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

characters=[' ', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', '/', '#']
max_label_length=96

class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, packed_input=False):
        super(ResidualLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.packed_input = packed_input
        
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size*self.num_direction, input_size)

    def forward(self, input, input_length=0, hidden_state=None):
        # input shape: batch_size, sequence_length, feature_dim
        
        if self.packed_input:
            # unpack input for residual connection
            input_unpack, _ = nn.utils.rnn.pad_packed_sequence(input, batch_first=True)

        rnn_output, hidden_state = self.rnn(input, hidden_state)
        
        if self.packed_input:
            # unpack output
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        
        batch_size, seq_length = rnn_output.shape[:2]
    
        # project the output back to the input dimension
        proj_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(batch_size, seq_length, -1)
        
        # residual connection
        if self.packed_input:
            output = input_unpack + proj_output
            # pack output
            output = nn.utils.rnn.pack_padded_sequence(output, input_length, batch_first=True, enforce_sorted=False)
        else:
            output = input + proj_output
            
        return output, hidden_state
            
    
class DeepResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, packed_input=False):
        super(DeepResidualLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            self.layers.append(ResidualLSTM(input_size, hidden_size, bidirectional, packed_input))
    
    def forward(self, input, input_length=0, hidden_state=None):
        # input shape: batch_size, sequence_length, feature_dim
        output = input
        if hidden_state == None:
            hidden_state = [None]*self.num_layers
        for i in range(self.num_layers):
            output, layer_hidden_state = self.layers[i](output, input_length, hidden_state[i])
            hidden_state[i] = layer_hidden_state
        return output, hidden_state
class LAS(nn.Module):
    def __init__(self, MFCC_dim=257, num_target=len(characters), max_decoding_length=max_label_length):
        super(LAS, self).__init__()
        
        self.input_dim = MFCC_dim
        self.hidden_dim = 64
        self.num_target = num_target
        self.max_decoding_length = max_decoding_length
        
        # one additional character embedding for padded invalid labels
        # during decoding, this entry might be used in teacher forcing
        # during testing, this will never be generated
        self.character_embedding = nn.Embedding(self.num_target+1, self.hidden_dim)
        
        # encoder LSTM
        self.encoder_LSTM = DeepResidualLSTM(self.input_dim, self.hidden_dim, num_layers=2, 
                                             bidirectional=True, packed_input=True)
        
        # decoder LSTM
        self.decoder_LSTM = DeepResidualLSTM(self.hidden_dim+self.input_dim, self.hidden_dim, 
                                             num_layers=2, bidirectional=False, packed_input=False)
        
        # FC layers for matching the dimension in attention
        self.decoder_FC = nn.Linear(self.hidden_dim+self.input_dim, self.input_dim)
        
        # output MLP
        # BOS and the padded invalid label will not be in the final output
        self.decoder_prob = nn.Sequential(nn.Linear(self.hidden_dim+self.input_dim*2, self.hidden_dim),
                                          nn.Tanh(),
                                          nn.Linear(self.hidden_dim, self.num_target-1)  
                                         )
        
    def encoding(self, input_feature, input_length):
        # input_feature: MFCC of shape (batch_size, max_frame, MFCC_dim)
        # input_length: denotes the actual length for each utterance in the batch, shape (batch_size,)
        # input label: oracle target labels of shape (batch_size, max_frame)
        
        # encoder
        
        enc_output, _ = self.encoder_LSTM(input_feature, input_length)
        
        # unpack the output
        enc_output, _ = nn.utils.rnn.pad_packed_sequence(enc_output, batch_first=True)
        
        return enc_output
    
    def decoding(self, enc_output, context_vector, prev_label, 
                 input_length, current_decoder_state=None):
        
        batch_size = enc_output.shape[0]
        
        # extract character embedding
        prev_label = self.character_embedding(prev_label)

        # step 1: calculate decoder RNN output
        decoder_input = torch.cat([prev_label, context_vector], 1)

        # pass it to decoder RNN
        current_decoder_output, current_decoder_state = self.decoder_LSTM(decoder_input.unsqueeze(1), 
                                                                          hidden_state=current_decoder_state)

        # step 2: update the attention context vector
        this_embedding = self.decoder_FC(current_decoder_output.squeeze(1))
        similarity = enc_output.bmm(this_embedding.unsqueeze(2)) 
        attention_weight = [F.softmax(similarity[i, :input_length[i]], dim=0).view(-1,1) for i in range(batch_size)]
        context_vector = [(enc_output[i,:input_length[i]] * attention_weight[i]).sum(0).unsqueeze(0) for i in range(batch_size)]
        context_vector = torch.cat(context_vector, 0)

        # step 3: concatenate the new attention context vector with the decoder output
        prob_input = torch.cat([current_decoder_output.squeeze(1), context_vector], 1)
        current_decoder_prob = self.decoder_prob(prob_input)

        return current_decoder_prob, current_decoder_state, context_vector

    def forward(self, input_feature, input_length, input_target_label=None, teacher_forcing=True):
        
        # encoding
        enc_output = self.encoding(input_feature, input_length)
        batch_size = enc_output.shape[0]
        
        decoder_output_prob = []
        decoder_output_label = []
        
        for step in range(self.max_decoding_length):
            if step == 0:
                # use BOS at the beginning
                # BOS is the last entry in the character list, second last entry in the character embeddings
                prev_label = torch.ones(batch_size).long() * (self.num_target - 1)
                context_vector = torch.zeros(batch_size, self.input_dim)
                current_decoder_state = None
            else:
                if teacher_forcing:
                    # apply teacher forcing
                    prev_label = input_target_label[:,step-1]
                else:
                    prev_label = decoder_output_label[-1].squeeze(1)
                        
            # decoding
            current_decoder_prob, current_decoder_state, context_vector = self.decoding(enc_output, context_vector, 
                                                                                        prev_label, input_length,
                                                                                        current_decoder_state)
            
            decoder_output_prob.append(current_decoder_prob.unsqueeze(1))
            _, sample_character_index = torch.max(current_decoder_prob, dim=1)  # select the one with the highest probability
            decoder_output_label.append(sample_character_index.unsqueeze(1))

        decoder_output_prob = torch.cat(decoder_output_prob, 1)  # batch, max_length, len(characters)-1 (no BOS)
        decoder_output_label = torch.cat(decoder_output_label, 1)  # batch, max_length

        return decoder_output_prob, decoder_output_label
        
    
# CE loss
def CE(output, target):
    # output shape: (batch, max_length, num_target)
    # target shape: (batch, max_length)
    
    batch_size, max_length, num_target = output.shape
    
    loss = nn.CrossEntropyLoss(ignore_index=len(characters))
    
    return loss(output.view(-1, num_target), target.view(-1,))

class dataset_pipeline(Dataset):
    def __init__(self, data, label_index, label_character, label_length, validation=False):
        super(dataset_pipeline, self).__init__()
        
        self.data = data
        self.label_index = label_index
        self.validation = validation
        self.label_length = label_length
        if self.validation:
            self.label_character = label_character
        
        self._len = data.shape[0]  # number of utterances
    
    def __getitem__(self, index):
        MFCC = torch.from_numpy(self.data[index,:,:].T).type(torch.float)
        MFCC_length = self.data.shape[2]
        label_index = torch.from_numpy(np.array(self.label_index[index])).long()
        if self.validation:
            label_character = self.label_character[index]
            return MFCC, MFCC_length, label_index, label_character
        else:
            label_length = self.label_length[index]
            return MFCC, MFCC_length, label_index, label_length
    
    def __len__(self):
        return self._len
    
    
# training and validation pipeline

def train(train_loader, model, optimizer ,epoch,log_step, versatile=True):
    start_time = time.time()
    model = model.train()
    train_loss = 0.
    
    # load batch data
    for batch_idx, data in enumerate(train_loader):
        MFCC, MFCC_length, label_index, label_length = data
        # pack the input batch
        MFCC_packed = nn.utils.rnn.pack_padded_sequence(MFCC, MFCC_length, 
                                                        batch_first=True, enforce_sorted=False)
        
        optimizer.zero_grad()

        len(MFCC_packed)
        # apply teacher forcing
        decoder_output_prob, decoder_output_label = model(MFCC_packed, MFCC_length, label_index)
        
        # CE as objective
        loss = CE(decoder_output_prob, label_index)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        
        train_loss += loss.data.item()
        
        #print(loss.data.item())
        
        if versatile:
            if (batch_idx+1) % log_step == 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | CE {:5.4f} |'.format(
                    epoch, batch_idx+1, len(train_loader),
                    elapsed * 1000 / (batch_idx+1), 
                    train_loss / (batch_idx+1)
                    ))
    
    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | CE {:5.4f} |'.format(
            epoch, (time.time() - start_time), train_loss))
    
    return train_loss

def validate(model, epoch):
    start_time = time.time()
    model = model.eval()
    
    all_decoder_output_label = []
    all_label_character = []
    
    # load batch data
    for batch_idx, data in enumerate(validation_loader):
        MFCC, MFCC_length, label_index, label_character = data
        # pack the input batch
        MFCC_packed = nn.utils.rnn.pack_padded_sequence(MFCC, MFCC_length, 
                                                        batch_first=True, enforce_sorted=False)
        
        with torch.no_grad():
            # no teacher forcing
            decoder_output_prob, decoder_output_label = model(MFCC_packed, MFCC_length, teacher_forcing=False)
            
            decoder_output_label = decoder_output_label.data.numpy()
            
            for batch in range(decoder_output_label.shape[0]):
                this_decoder_output_label = decoder_output_label[batch]
                EOS_location = np.where(this_decoder_output_label == (len(characters) - 2))[0]
                if len(EOS_location) > 0:
                    this_decoder_output_label = this_decoder_output_label[:EOS_location[0]]
                all_decoder_output_label.append(this_decoder_output_label)
                all_label_character.append(label_character[batch][:-1])
        
    # calculate CER and WER on the entire training set
    num_utterance = len(all_decoder_output_label)
    
    prediction_list = [[characters[all_decoder_output_label[j][i]] for i in range(len(all_decoder_output_label[j]))] 
                       for j in range(num_utterance)]
    prediction_list = [''.join(prediction_list[i]) for i in range(num_utterance)]

    all_cer = fastwer.score(prediction_list, all_label_character, char_level=True)
    all_wer = fastwer.score(prediction_list, all_label_character)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | Corpus-level CER: {:.2f}% | Corpus-level WER: {:.2f}% |'.format(
        epoch, (time.time() - start_time), all_cer, all_wer))
    
    return all_cer