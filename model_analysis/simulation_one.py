import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import sys
sys.path.append('/home3/ebrahim/isr/')
from datasets import OneHotLetters
from torch.utils.data import DataLoader
torch.set_num_threads(2)
import pandas as pd
import seaborn as sns
import os
import pickle
from model import LSTMCELL, RNN_feedback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rn', type=str, 
                    help="run number corresponding to model used for analyses")
parser.add_argument('--folder', type=str, default='simulation_one_weights/',
                    help="Folder where model is saved")
parser.add_argument('--save_folder', type=str, default='simulation_one_weights/',
                    help="Folder to save output")
                    

args = parser.parse_args()

run_number = args.rn
model_folder = args.folder
save_folder = args.save_folder


class simulation_one():

    def __init__(self, model, save_path):

        self.model = model
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # for figure 6
        self.ppr_list = []
        self.list_length = []

    def run_model(self, device, dataloader, test_list_length):

        self.model.to(device)

        y_hat_all = []

        # generate model_output
        for X_test, y_test in dataloader:

            X_test = X_test.to(device)
            y_test = y_test.to(device)

            x, r, y_hat = self.model.init_states(X_test.shape[0], device)

            model.eval()

            with torch.no_grad():
                # iterate to 2nd to last input (b/c ignoring end of list cue)
                for timestep in range(X_test.shape[1]-1):
                    y_hat, x, r = self.model(X_test[:, timestep, :], x, r, y_hat)
                    if timestep >= test_list_length:
                        y_hat_all.append(y_hat)

        # pass data to figures_data() function
        # move to cpu so that numpy transfer is possible 
        ppr, transpositions = self.figure_data(y_test[:, test_list_length:-1, :].cpu(), 
        torch.stack(y_hat_all, axis=1).cpu(), test_list_length)

        self.ppr_list.append(ppr)
        self.list_length.append(test_list_length)

        if test_list_length == 6:
            self.figure_seven_plot(transpositions)
            
        if len(self.ppr_list) == 6:
            self.figure_six_plot()

    def figure_data(self, y_test, y_hat, test_list_length):

        '''
        @param y_test (tensor): target output, shape (trials, timesteps, classes)
        @param y_hat (tensor): model output (after softmax), shape (trials, timesteps, classes)
        '''

        predictions = y_hat[:, :, :-1].argmax(2) # class with highest probability
        targets = y_test.argmax(2)
        num_trials = y_test.shape[0]

        # records index where model output at a given timestep and trial
        # equals target output
        transpositions = np.empty((targets.shape[1], targets.shape[0])) 
        
        repitition_errors = 0
        repitition_trials = np.zeros(num_trials)
        repitition_distances = []
        transposition_errors = 0 

        # porportion of lists perfectly recalled, one line ;) 
        ppr = torch.all(torch.eq(predictions, targets), dim=1).sum() / num_trials

        if test_list_length != 6:
            return ppr, None

        print("PPR: ", ppr)

        for trial in range(y_test.shape[0]):
            for timestep in range(y_test.shape[1]):
                # if a target item is found more than once in the predicted list
                # increment number of repitition errors by amount of excess repeats
                matched_tp = np.argwhere(targets[trial, timestep] == predictions[trial]).numpy().squeeze()
                if matched_tp.size > 1:
                    repitition_errors += matched_tp.size - 1
                    repitition_trials[trial] = 1
                    # distance between repeated elements in predictions
                    for dist in np.ediff1d(matched_tp):
                        repitition_distances.append(dist)

                # locate position where predicted letter equals targets
                # if correct, matches + 1 should equal timestep
                matched_pt = np.argwhere((predictions[trial, timestep] == targets[trial]).numpy())

                if matched_pt.size == 1:
                    transpositions[timestep, trial] = int(matched_pt) + 1

                    if int(matched_pt) != timestep:
                        transposition_errors += 1 
        
        # total errors 
        print("mean repitition distance: ", np.mean(repitition_distances)) 
        print("Fraction of trials with repititions: ", np.sum(repitition_trials)/num_trials)
        print("Fraction of errors that are repititions: ", (np.sum(repitition_trials)/num_trials)/(1-ppr.item()))

        return ppr, transpositions 

    def figure_six_plot(self):

        plt.plot(self.list_length, self.ppr_list, marker='o')
        plt.xlabel("List lengths")
        plt.ylabel("Accuracy")
        plt.savefig(self.save_path + 'fig6', dpi=400, bbox_inches='tight')

    def figure_seven_plot(self, transpositions):

        fig, ax = plt.subplots(1,6, sharex=False, sharey=True, figsize=(12,8))
        sns.despine()
        for i in range(6): 
            ax[i].hist(x = transpositions[i, :], bins=[1,2,3,4,5,6,7], density=True, color=(0,0,0), rwidth=.8)
            ax[i].set_xlabel(i+1, fontsize=18)
            ax[i].set_xticks([])
            ax[i].set_yticks([0, .25, .5, .75, 1])
            ax[0].tick_params(axis='y', which='major', labelsize=18)
            if i > 0:
                ax[i].get_yaxis().set_visible(False)
                sns.despine(left=True, ax=ax[i])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.text(-15, -.1, 'Position', fontsize=24)
        plt.savefig(self.save_path + 'fig7', dpi=400, bbox_inches='tight')
        plt.close()

###################### run code ###########################
base = '/home3/ebrahim/isr/'
path = base + 'saved_models/' + model_folder + 'run_' + run_number + '/'

with open(path + 'model_settings.pkl', 'rb') as handle:
    ms = pickle.load(handle)
print(ms)
model = RNN_feedback(ms['is'], ms['hs'], ms['os'], ms['init_wrec'], ms['alpha_s'], 
ms['alpha_r'], ms['feedback_scaling'])

if run_number == '13':
    model.load_state_dict(torch.load(path + '29_model_weights.pth'))
else:
    model.load_state_dict(torch.load(path + 'final_model_weights.pth'))

device = torch.device("cuda:0")
test_size = 5000
hidden_size = 200
num_letters = 10
input_size = output_size = num_letters + 1
test_list_length = 6
save_path = 'saved_data/' + save_folder + 'run_' + run_number + '/'

sim_one = simulation_one(model, save_path)

for ll in range(4, 10, 1):
    test_dataloader = DataLoader(OneHotLetters(ll, None, num_letters=num_letters,
                                    test_mode=True, num_test_trials=test_size), batch_size=test_size, shuffle=False)
    sim_one.run_model(device, test_dataloader, ll)


