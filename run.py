from logging import logProcesses
import os
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from model import RNN_feedback
from datasets import OneHotLetters
torch.set_num_threads(4)

# generate training + testing data
list_length = 9 # length of longest list 
test_list_length = 6 # only test on lists of length 6
num_cycles = 200000
train_batch = 1
test_size = 5000
num_letters = 26
train_dataloader= DataLoader(OneHotLetters(list_length, num_cycles, num_letters=num_letters,
                                test_mode=False), batch_size=train_batch, shuffle=False)
test_dataloader = DataLoader(OneHotLetters(test_list_length, None, num_letters=num_letters, 
                                test_mode=True), batch_size=test_size, shuffle=False)

# training details 
loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
lr=0.001

# init model
input_size = output_size = num_letters + 1
hidden_size = 200

device = torch.device("cuda:0")

loss_list = []
accuracy_list = []

def train_loop(save_path, checkpoint_epoch=10000):

    model = RNN_feedback(input_size, hidden_size, output_size)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    os.makedirs(save_path, exist_ok=True)

    save_number = 0
    model
    model.train()
    loss_per_1000 = 0.0

    for batch_idx, (X, y) in enumerate(train_dataloader):

        X = X.to(device)
        y = y.to(device)

        ll = (batch_idx % list_length) + 1 # list length for current batch

        # Compute prediction and loss
        hidden, y0 = model.init_hidden_output_state(train_batch, device)
        loss = 0.0

        # run RNN and compute loss
        # perform teacher forcing for o2h layer
        for timestep in range(X.shape[1]):
            if timestep == 0:
                y_hat, hidden = model(X[:, timestep, :], hidden, y0)
            else:
                y_hat, hidden = model(X[:, timestep, :], hidden, y[:, timestep-1, :])

            loss += loss_fn(y_hat.log(), y[:, timestep, :])
            loss_per_1000 += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print model loss every 1000 trials 
        if batch_idx % 1000 == 0:
            print("Batch number: ", batch_idx)
            loss_per_1000 /= 1000
            if batch_idx != 0:
                loss_list.append(round(loss_per_1000.item(), 5))
            print("Loss: ", round(loss_per_1000.item(), 5))
            loss_per_1000 = 0.0

        if batch_idx % checkpoint_epoch == 0:
            # check accuracy every checkpoint_epoch trials and save model
            save_model_path = save_path + str(save_number) + '_'
            accuracy = checkpoint(save_model_path, model)
            save_number += 1
            accuracy_list.append(accuracy)
            
        if accuracy > .58:
            print("TRAINING IS DONE :)")
            np.save(save_path + 'accuracy_list', accuracy_list)
            np.save(save_path + 'loss_list', loss_list)
            break 

def checkpoint(save_path, model):

            '''
            This function computes model accuracy on lists of a fixed size.
            Unlike training, no teacher forcing is performed from o2h layer. 
            In other words, y_hat (not y) from previous timestep is used.
            Accuracy is computed only on the recall portion of the list,
            ignoring the end of list cue. Accuracy indicates the proportion 
            of lists perfectly recalled. 

            @param save_path (str): where to save model and accuracy
            @param model (pytorch model): RNN used to generate predictions 
            '''

            ppr = np.zeros(test_size) # proportion perfect_recall
            model.eval()
            y_hat_all = []

            for X_test, y_test in test_dataloader:

                X_test = X_test.to(device)
                y_test = y_test.to(device)
                hidden, y_hat = model.init_hidden_output_state(test_size, device)

                with torch.no_grad():
                    for timestep in range(X_test.shape[1]):
                        y_hat, hidden = model(X_test[:, timestep, :], hidden, y_hat)
                        # recall phase
                        if timestep >= test_list_length:
                            y_hat_all.append(y_hat)
                    accuracy = compute_accuracy(torch.stack(y_hat_all, axis=1), y_test[:, test_list_length:, :])

            model.train()
            
            # save model and accuracy
            torch.save(model, save_path + 'model.pth')
            print("Accuracy: ", accuracy.item())
            return accuracy.item()


def compute_accuracy(y_hat, y_test):

    '''
    Computes the fraction of lists that were perfectly recalled. 
    @param y_hat (Tensor): model predictions for all batches
    @param y_test (Tensor): target outputs 
    '''
    predictions = y_hat[:, :, :-1].argmax(2) # batch_size x recall timesteps (ignoring end of list item)
    targets = y_test.argmax(2) # batch_size x recall timesteps
    return torch.all(torch.eq(predictions, targets), dim=1).sum() / y_test.shape[0]

save_path = 'saved_models/simulation_one/'
train_loop(save_path + 'run_7/')


 




    








    



