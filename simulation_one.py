import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
from datasets import OneHotLetters
from torch.utils.data import DataLoader
torch.set_num_threads(2)
import pandas as pd
import seaborn as sns
import os
import pickle
from RNNcell import RNN_one_layer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rn', type=str, 
                    help="run number corresponding to model used for analyses")

args = parser.parse_args()
run_number = args.rn

class simulation_one():

    def __init__(self, model, num_trials, max_length):

        '''
        @param model: PyTorch model 
        @param num_trials: number of trials used to test model
        @param max_length: longest list the model is tested on 
        '''

        self.model = model
        self.num_trials = num_trials
        self.max_length = max_length
        self.met_accuracy = False
        self.ppr_list = [] # Proportion perfectly recalled
        self.ppr_six = 0 # ppr on lists of length six 
        self.list_length = []
        self.ARD = 0 # Average repitition distance
        self.R_T = 0 # Fraction of transposition errors that are repititions 
        self.relative_error_ratio_list = [] # Ratio of relative errors

    def run_model(self, device, dataloader, test_list_length):

        self.model.to(device)

        y_hat_all = []

        # generate model_output
        for X_test, y_test in dataloader:

            X_test = X_test.to(device)
            y_test = y_test.to(device)

            y_hat, h = self.model.init_states(X_test.shape[0], device)

            self.model.eval()

            with torch.no_grad():

                # iterate to 2nd to last input (b/c ignoring end of list cue)
                for timestep in range(X_test.shape[1]-1):
                    y_hat, h = self.model(X_test[:, timestep, :], h, y_hat)

                    y_hat = torch.softmax(y_hat, dim=1)

                    if timestep >= test_list_length:
                        y_hat_all.append(y_hat)

        self.y_test_recall = y_test[:, test_list_length:-1, :].cpu().argmax(2)
        self.y_hat_recall = torch.stack(y_hat_all, axis=1).cpu().argmax(2)

        ppr = self.compute_ppr()

        if test_list_length >= 6:
            transpositions, ARD, R_T, relative_error_ratio = self.transposition_matrix(test_list_length)

            if test_list_length == 6:
                self.ARD = ARD
                self.R_T = R_T
                self.ppr_six = ppr
                self.transpositions_six = transpositions
            
            if test_list_length > 6:
                self.relative_error_ratio_list.append(relative_error_ratio)
        
        self.ppr_list.append(ppr)
        self.list_length.append(test_list_length)

    def compute_ppr(self):

        # porportion of lists perfectly recalled
        ppr = torch.all(torch.eq(self.y_hat_recall, self.y_test_recall), dim=1).sum() \
        / self.num_trials

        return ppr 

    def transposition_matrix(self, test_list_length):

        # records index where model output at a given timestep and trial
        # equals target output
        transpositions = np.empty((self.y_test_recall.shape[1], 
        self.y_hat_recall.shape[0])) 

        repitition_errors = 0
        repitition_trials = np.zeros(self.num_trials)
        repitition_distances = []
        transposition_errors = 0 
        relative_errors = 0
        non_relative_errors = 0
    
        for trial in range(self.num_trials):
            for timestep in range(test_list_length):

                matched_tp = np.argwhere(self.y_test_recall[trial, timestep] == \
                self.y_hat_recall[trial]).numpy().squeeze()

                # if a target item is found more than once in the predicted list
                # increment number of repitition errors by amount of excess repeats
                if matched_tp.size > 1:

                    repitition_errors += matched_tp.size - 1
                    repitition_trials[trial] = 1

                    # distance between repeated elements in predictions
                    for dist in np.ediff1d(matched_tp):
                        repitition_distances.append(dist)

                # locate position where predicted letter equals targets
                matched_pt = np.argwhere((self.y_hat_recall[trial, timestep] == \
                self.y_test_recall[trial]).numpy())

                # ensure there is a match (i.e. no intrusion error)
                if matched_pt.size == 1:
                    transpositions[timestep, trial] = int(matched_pt) + 1

                    # if recalled letter isn't at correct position, then increment
                    # number of transposition errors
                    if int(matched_pt) != timestep:
                        transposition_errors += 1 

                        # Check for relative error 
                        if timestep != test_list_length-1:

                            # timestep where next predicted letter equals to target list
                            matched_pt_next = np.argwhere((self.y_hat_recall[trial, timestep+1] == \
                            self.y_test_recall[trial]).numpy())
                            
                            # if next predicted letter is found in target list
                            if matched_pt_next.size==1:

                                # if it is found in the position right after, then we have a relative error
                                if matched_pt_next - matched_pt == 1:
                                    relative_errors += 1            

                                # otherwise, no relative_error
                                else:
                                    non_relative_errors += 1
        
        ARD = round(np.mean(repitition_distances),3)

        R_T = round(repitition_errors/transposition_errors, 3)
 
        relative_error_ratio = round(relative_errors/(non_relative_errors+relative_errors), 3)

        return transpositions, ARD, R_T, relative_error_ratio


    def figure_six_plot(self, wandb):
        fig, ax = plt.subplots(1,1)
        ax.plot(self.list_length, self.ppr_list, marker='o')
        ax.set_xlabel("List lengths")
        ax.set_ylabel("Accuracy")
        wandb.log({'ppr_plot': fig})
        plt.close()

    def figure_seven_plot(self, wandb):

        fig, ax = plt.subplots(1,6, sharex=False, sharey=True, figsize=(12,8))
        sns.despine()
        for i in range(6): 
            ax[i].hist(x = self.transpositions_six[i, :], bins=[1,2,3,4,5,6,7], density=True, color=(0,0,0), rwidth=.8)
            ax[i].set_xlabel(i+1, fontsize=18)
            ax[i].set_xticks([])
            ax[i].set_yticks([0, .25, .5, .75, 1])
            ax[0].tick_params(axis='y', which='major', labelsize=18)
            if i > 0:
                ax[i].get_yaxis().set_visible(False)
                sns.despine(left=True, ax=ax[i])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.text(-15, -.1, 'Position', fontsize=24)
        wandb.log({'transposition_plot': wandb.Image(fig)})
        plt.close()

    def save_metrics(self, wandb):

        wandb.log({'ARD':self.ARD, 'R/T': self.R_T, 'ppr_six': self.ppr_six, 
        'relative_error_ratios': np.mean(self.relative_error_ratio_list)})



