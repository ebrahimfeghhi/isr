from logging import logProcesses
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datasets import OneHotLetters
import pickle
import sys
sys.path.append('/home3/ebrahim/isr/models/')
from RNNcell import RNN_one_layer, RNN_two_layers
torch.set_num_threads(4)
import wandb

# generate training + testing data
list_length = 9 # length of longest list 
test_list_length = 6 # only test on lists of length 6
num_cycles = 200000
train_batch = 1
test_size = 5000
num_letters = 10
hs = 200
loss_func = 'ce'
lr = 0.001
stopping_criteria = .58
feedback_scaling = 1.0
opt = 'SGD'
mt = 1
nonlin = 'sigmoid'
clipping = False
clip_factor_arr = 10
eps = 1e-7
fb_type = 0 

# init model
input_size = output_size = num_letters + 1
device = torch.device("cuda:0")

'''
wandb.init(project="my-test-project", config={'is':input_size, 'hs':hs, 'os':output_size, 'lr':lr, 'loss_func': loss_func,
    'optimizer':opt, 'stopping_criteria':stopping_criteria, 'feedback_scaling':feedback_scaling, 
    'nonlin': nonlin, 'clipping': clipping, 'clip_factor': clip_factor, 'list_length': list_length, 
    'test_list_length': test_list_length})
'''

def train_loop(save_path, clip_factor, checkpoint_epoch=10000):


    model_settings = {'is':input_size, 'hs':hs, 'os':output_size, 'lr':lr, 'loss_func': loss_func,
    'optimizer':opt, 'stopping_criteria':stopping_criteria, 'feedback_scaling':feedback_scaling, 
    'nonlin': nonlin, 'clipping': clipping, 'clip_factor': clip_factor, 'list_length': list_length, 
    'test_list_length': test_list_length, 'fb_type':fb_type}

    with open(save_path + 'model_settings.pkl', 'wb') as f:
        pickle.dump(model_settings, f)

    loss_list = []
    accuracy_list = []

    train_dataloader= DataLoader(OneHotLetters(list_length, num_cycles, num_letters=num_letters,
                                test_mode=False), batch_size=train_batch, shuffle=False)
    test_dataloader = DataLoader(OneHotLetters(test_list_length, None, num_letters=num_letters,     
                                test_mode=True), batch_size=test_size, shuffle=False)


    loss_fn = torch.nn.CrossEntropyLoss()

    if mt == 1:
        model = RNN_one_layer(input_size, hs, output_size, feedback_scaling, nonlin)
    else:
        model = RNN_two_layers(input_size, hs, output_size, feedback_scaling, nonlin, fb_type)

    model = model.to(device)

    #wandb.watch(model, log='all', log_freq=10000)
    
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    os.makedirs(save_path, exist_ok=True)

    save_number = 0
    model
    model.train()
    loss_per_1000 = 0.0

    for batch_idx, (X, y) in enumerate(train_dataloader):

        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        y0, h0 = model.init_states(train_batch, device)
        
        loss = 0.0

        # run RNN and compute loss
        # perform teacher forcing for o2h layer (pass y instead of y_hat)
        for timestep in range(X.shape[1]):

            if timestep == 0:
                y_hat, h = model(X[:, timestep, :], h0, y0)
            else:
                y_hat, h = model(X[:, timestep, :], h, y[:, timestep-1, :])

            loss += loss_fn(y_hat, y[:, timestep, :])
            loss_per_1000 += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # ensure that norm of all gradients falls under clip_factor
        if clipping: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_factor)

        optimizer.step()

        #wandb.log({'loss': loss})

        # print model loss every 1000 trials 
        if batch_idx % 1000 == 0 and batch_idx != 0:
            print("Batch number: ", batch_idx)
            loss_per_1000 /= 1000
            if batch_idx != 0:
                loss_list.append(round(loss_per_1000.item(), 5))
            print("Loss: ", round(loss_per_1000.item(), 5))
            loss_per_1000 = 0.0

        if batch_idx % checkpoint_epoch == 0:
            # check accuracy every checkpoint_epoch trials and save model
            save_model_path = save_path + str(save_number) + '_'
            accuracy = checkpoint(save_model_path, model, test_dataloader)
            save_number += 1
            accuracy_list.append(accuracy)
            
        if accuracy > stopping_criteria or torch.isnan(loss):
            print("TRAINING IS DONE :)")
            torch.save(model.state_dict(), save_path + 'final_model_weights.pth')
            np.save(save_path + 'accuracy_list', accuracy_list)

            # append loss list with nan if training ends due to nan loss
            if torch.isnan(loss):
                np.save(save_path + 'loss_list_nan', loss_list)
            else:
                np.save(save_path + 'loss_list', loss_list)
            break 

def checkpoint(save_path, model, test_dataloader):

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
            model.eval()
            y_hat_all = []

            for X_test, y_test in test_dataloader:

                X_test = X_test.to(device)
                y_test = y_test.to(device)

                y_hat, h = model.init_states(test_size, device)

                with torch.no_grad():
                    # iterate to 2nd to last input (b/c ignoring end of list cue)
                    for timestep in range(X_test.shape[1]-1):
                        y_hat, h = model(X_test[:, timestep, :], h, y_hat)

                        y_hat = torch.softmax(y_hat, dim=1)

                        # recall phase
                        if timestep >= test_list_length:
                            y_hat_all.append(y_hat)

                    accuracy = compute_accuracy(torch.stack(y_hat_all, axis=1), y_test[:, test_list_length:-1, :])

            model.train()
            
            # save model and accuracy
            torch.save(model.state_dict(), save_path + 'model_weights.pth')
            print("Accuracy: ", accuracy.item())
            return accuracy.item()

def compute_accuracy(y_hat, y_test):

    '''
    Computes the fraction of lists that were perfectly recalled. 
    @param y_hat (Tensor): model predictions for all batches
    @param y_test (Tensor): target outputs 
    '''
    predictions = y_hat[:, :, :-1].argmax(2) # batch_size x recall timesteps (ignoring end of list input)
    targets = y_test.argmax(2) # batch_size x recall timesteps
    return torch.all(torch.eq(predictions, targets), dim=1).sum() / y_test.shape[0]

rn = '29'
save_path = 'saved_models/simulation_one_cell/run_' + rn + '/'
os.makedirs(save_path, exist_ok=True)
train_loop(save_path)







    








    



