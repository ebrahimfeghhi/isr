from logging import logProcesses
import os
import torch 
from torch.utils.data import DataLoader
import numpy as np
from datasets import OneHotLetters
from RNNcell import RNN_one_layer
torch.set_num_threads(4)
import wandb
import argparse
from simulation_one import simulation_one

parser = argparse.ArgumentParser()
parser.add_argument('--rn', type=str, 
                    help="run number corresponding to model used for analyses")

args = parser.parse_args()

rn = args.rn

device = torch.device("cuda:0")

# generate training + testing data
training_settings = {
    'max_length' : 9, 
    'test_list_length': 6, 
    'num_cycles': 200000,
    'train_batch': 1,
    'test_size': 5000,
    'num_letters': 10,
    'hs': 200,
    'lr': 0.001,
    'stopping_criteria': .58,
    'feedback_scaling': 1.0,
    'opt': 'SGD',
    'nonlin': 'sigmoid',
    'clipping': False, 
    'clip_factor': 10}

def train_loop(save_path, checkpoint_epoch=10000):

    wandb.init(project="serial_recall_RNNs", config=training_settings)
    input_size = output_size = wandb.config['num_letters'] + 1

    loss_list = []

    train_dataloader= DataLoader(OneHotLetters(wandb.config['max_length'], wandb.config['num_cycles'], 
                                num_letters=wandb.config['num_letters'], test_mode=False), 
                                batch_size=wandb.config['train_batch'], shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()

    model = RNN_one_layer(input_size, wandb.config['hs'], 
    output_size, wandb.config['feedback_scaling'], wandb.config['nonlin'])

    model = model.to(device)

    wandb.watch(model, log='all', log_freq=10000)

    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config['lr'])

    os.makedirs(save_path, exist_ok=True)

    model.train()
    loss_per_1000 = 0.0

    for batch_idx, (X, y) in enumerate(train_dataloader):

        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        y0, h0 = model.init_states(wandb.config['train_batch'], device)
        
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
        if wandb.config['clipping']: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), wandb.config['clip_factor'])

        optimizer.step()

        wandb.log({'loss': loss})

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
            sim_one = simulation_one(model, save_path, wandb.config['test_size'], wandb.config['max_length'])
            sim_one = checkpoint(sim_one)
            wandb.log({'accuracy': sim_one.ppr_six})
            print("Accuracy: ", round(sim_one.ppr_six,5))

        if sim_one.met_accuracy == True or torch.isnan(loss):
            torch.save(model.state_dict(), save_path + 'final_model_weights.pth')
            sim_one.figure_six_plot(wandb)
            sim_one.figure_seven_plot(wandb)
            sim_one.save_metrics(wandb)
            break

def checkpoint(sim_one):
    
    for ll in range(4, wandb.config['max_length']+1, 1):

        test_dataloader = DataLoader(OneHotLetters(ll, None, num_letters=wandb.config['num_letters'],
                                    test_mode=True, num_test_trials=wandb.config['test_size']), 
                                    batch_size=wandb.config['test_size'], shuffle=False)
        sim_one.run_model(device, test_dataloader, ll)

        if ll == 6 and sim_one.ppr > wandb.config['stopping_criteria']:
            sim_one.met_accuracy = True 
        
        return sim_one

def main(rn):
    save_path = 'saved_models/simulation_one_cell/run_' + rn + '/'
    os.makedirs(save_path, exist_ok=True)
    
    train_loop(save_path)







    








    



