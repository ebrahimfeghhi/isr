import torch 
import torch.nn as nn
from torch import optim 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import RNN_feedback, RNN_pytorch, internet_RNN
from datasets import OneHotLetters

# generate training + testing data
sequence_length = 1
num_cycles = 200000 # a cycle is a set of trials from 1 to sequence_length 
train_batch = 1
training_data = DataLoader(OneHotLetters(sequence_length, num_cycles), batch_size=train_batch, 
                                        shuffle=False)
input_size = 26
hidden_size = 200
output_size = 26
model = RNN_feedback(input_size, hidden_size, output_size)

def train_loop(dataloader, model, lr=.0001, cuda_bool=True):

    device = torch.device("cuda:0" if cuda_bool else "cpu")
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        hidden = torch.zeros(X.size(0), hidden_size).to(device)
        output = torch.zeros(X.size(0), output_size).to(device)
        loss = 0.0

        for timestep in range(X.shape[1]):
            y_hat = model(X[:, timestep, :], hidden, output)
            loss += loss_fn(y_hat, y[:, timestep])
            if batch % 1000 == 0:
                print("Loss: ", loss.item())
                print("y, y_hat: ", y.item(), torch.argmax(y_hat, axis=1).item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_loop_pytorch(dataloader, model, lr=.001, cuda_bool=True):

    device = torch.device("cuda:0" if cuda_bool else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for batch, (X, y) in enumerate(dataloader):

        # shape (batch_size, sequence_length, number_classes) for both X and y
        X = X.to(device) 
        y = y.to(device)

        h0 = torch.zeros(1, X.size(0), hidden_size).to(device)
        out0 = torch.zeros(1, X.size(0), output_size).to(device)
        loss = 0.0
        
        # Compute prediction and loss
        y_hat = model(X, h0, out0)

        for timestep in range(sequence_length): 
            loss += loss_fn(y_hat[:, timestep, :], y[:, timestep])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            print("Loss: ", loss.item())
            print("y, y_hat: ", y.item(), torch.argmax(y_hat, axis=2).item())
            
       

train_loop(training_data, model)