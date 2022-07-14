import matplotlib
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from itertools import permutations 
import sys
base = '/home3/ebrahim/isr/'
sys.path.append(base)
from RNNcell import RNN_one_layer, RNN_two_layers
from RNN_feedback import RNN_feedback
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import os
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--rn', type=str, 
                    help="run number corresponding to model used for analyses")
parser.add_argument('--mt', type=int, default=1,
                    help="0 for RNN with feedback, 1 for one layer RNN, 2 for two layer RNN")
                                

args = parser.parse_args()
run_number = args.rn
mt = args.mt

class Regression_Trials(Dataset):

    def __init__(self, num_letters, letters_subset, list_length):

        '''
        Dataset is composed of all possible permutations of 
        the available subset of letters (s) for the specified list length (l).

        Example: If a,b, and c are the available letters, and list length is 2,
        then the dataset will be {ab, ba, ac, ca, bc, cb}.

        Number of trials generated is equal to s! / (s-l)!. 

        @param num_letters: number of total letters (classes - 1) RNN was trained on
        @param letters_subset: subset of letters used for regression analyses 
        @param list_length: length of list for regression analyses 
        '''
        
        X_p = [] # store permutations 
        X_int = permutations(letters_subset, list_length)

        for p in X_int:
            X_p.append(p)

        X_int = np.stack(X_p) # shape: num_permutations x list_length 
        
        recall_cue = np.ones((X_int.shape[0], list_length+1)) * num_letters 
        self.X = torch.nn.functional.one_hot(torch.from_numpy(np.hstack((X_int, recall_cue))).to(torch.long)
        , num_classes=num_letters+1)

        end_of_list_cue = np.ones((X_int.shape[0], 1)) * num_letters
        y_int = torch.from_numpy(np.hstack((X_int, X_int, end_of_list_cue))).to(torch.long)
        self.y = torch.nn.functional.one_hot(y_int, num_classes=num_letters+1)

        self.X_reg = np.hstack((X_int, X_int)).T

    def __len__(self):

        return self.X.size(0)

    def __getitem__(self, idx):

        return self.X[idx].to(torch.float32), self.y[idx].to(torch.float32)

def inverse_sigmoid(y):

    '''
    Linearizes model activity 
    '''
    return torch.log(y/(1-y))

def regression_RNN_activity(model, dataloader, batch_size, loss_func):

    model.eval()

    hidden_arr = []

    with torch.no_grad():
        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)

            if mt == 0:
                x, h, y_hat = model.init_states(X.shape[0], device)
            else: 
                y_hat, h = model.init_states(X.shape[0], device)

            model.eval()
            with torch.no_grad():
                for timestep in range(X.shape[1]-1):
                    if mt == 0:
                        y_hat, x, h = model(X[:, timestep, :], x, h, y_hat)
                    else:
                        y_hat, h = model(X[:, timestep, :], h, y_hat)

                    if loss_func == 'ce':
                        y_hat = torch.softmax(y_hat, dim=1)

                    if nonlin == 'linear':
                        hidden_arr.append(h)
                    elif nonlin == 'sigmoid':
                        if mt == 2:
                            hidden_arr.append(inverse_sigmoid(h[1]))
                        else: 
                            hidden_arr.append(inverse_sigmoid(h))
            
    return torch.stack(hidden_arr)

def compute_element_vectors(X_all, y_all, num_features, timestep, seed):

    '''
    @param X_all (numpy array): shape num_permutations x (num_features x num_timesteps)
    @param y_all (numpy array): shape num_timesteps x num_permutations x num
    @param num_features (int): number of features used to encode each letters
    @param timestep (int): timestep to predict y, given X from t = 0:timestep
    @param seed (int): seed used to initalize random number generator

    Ridge regression model is fit from selected portions of X to y[timestep],
    and columns of weights correspond to element vectors. 
    '''
    
    X = X_all[:, :num_features*timestep]
    y = y_all[timestep-1]
    
    rng = np.random.default_rng(seed)
    train_ind = rng.choice(X.shape[0], int(X.shape[0]*.8))
    test_ind = np.setdiff1d(np.arange(0,X.shape[0],1), train_ind) 

    reg = Ridge(alpha=.01).fit(X[train_ind], y[train_ind])
    y_hat = reg.predict(X[test_ind])

    r2_score_value = r2_score(y[test_ind], y_hat)

    return round(r2_score_value,5), reg.coef_

def partition_W(W, num_features, MA):
    for i in range(int(W.shape[1]/num_features)):
        MA[i].append(W[:, i*num_features:(i+1)*num_features])


def positional_similarity_simple(t1, t2, hidden_activity, X):

    # obtain input and hidden activity at time t1 and t2
    Xt1 = X[:, t1]
    Xt2 = X[:, t2]
    ht1 = hidden_activity[t1]
    ht2 = hidden_activity[t2]

    # list of length number of unique letters in X
    # for every unique letter, stores the model representations at 
    # respective times 
    item_representations_t1 = []
    item_representations_t2 = []

    # Step 1) Obtain model representation for each item at t1 and t2
    for x in np.unique(X):

        # locate all instances where a given letter occurs at t1 and t2
        x_location_t1 = np.argwhere(Xt1 == x)
        x_location_t2 = np.argwhere(Xt2 == x)

        # append or average across these representations 
        item_representations_t1.append(np.ravel(ht1[x_location_t1]))
        item_representations_t2.append(np.ravel(ht2[x_location_t2]))

    corr_same = []
    corr_diff = []

    # Step 2) Compute similarity between item representations 
    for i, ht1 in enumerate(item_representations_t1):
        for j, ht2 in enumerate(item_representations_t2):
            if i == j:
                r_same, _ = pearsonr(ht1, ht2)
                corr_same.append(r_same)
            else:
                r_diff, _ = pearsonr(ht1, ht2)
                corr_diff.append(r_diff)

    return corr_same, corr_diff

def figure_4_code(W_encoding, ls):

    '''
    @param W_encoding: weights of linear regression model at final encoding step 
    @param ls: length of letters subset list 
    '''

    df = pd.DataFrame(W_encoding)
    W_corr = df.corr(method='pearson')
    sns.heatmap(W_corr)
    plt.title("Correlation between element vectors")
    plt.savefig(save_path + 'fig_4_ev_corrmap', dpi=400, bbox_inches='tight')
    plt.close()

    c1 = []
    c2 = []
    c3 = []


    for i in range(ls):
        c1.append(W_corr.iloc[i, i+ls]) # same item one position apart
        c2.append(W_corr.iloc[i, i+2*ls]) # same item two positions apart
        c3.append(W_corr.iloc[i, i+3*ls]) # same item three positions apart

    c0_n = []
    c1_n = []
    c2_n = []
    c3_n = []

    for i in range(ls):
        for j in range(ls):
            if i == j:
                continue
            c0_n.append(W_corr.iloc[i,j]) # distinct items at same position
            c1_n.append(W_corr.iloc[i,j+ls]) # distinct items one position apart
            c2_n.append(W_corr.iloc[i,j+2*ls]) # distinct items two positions apart
            c3_n.append(W_corr.iloc[i,j+3*ls]) # distinct items three positions apart

    c_same = [c1, c2, c3]
    c_diff = [c0_n, c1_n, c2_n, c3_n]


    np.save(save_path + 'positional_corr_same', np.stack(c_same))
    np.save(save_path + 'positional_corr_diff', np.stack(c_diff))

def figure_5_code(MA):

    labels = ['1', '2', '3', '4']
    for p in range(list_length):
        MA_t = np.stack(MA[p])
        cosine_sim = np.zeros((MA_t.shape[0], MA_t.shape[2]))
        for t in range(cosine_sim.shape[0]):
            for l in range(cosine_sim.shape[1]):
                cosine_sim[t,l] = np.dot(MA_t[t, :, l], h2o_weight_subset[l]
                ) / (np.linalg.norm(MA_t[t, :, l]) * np.linalg.norm(h2o_weight_subset[l]))
        plt.plot(np.arange(p+1,int(list_length*2)+1, 1), np.mean(cosine_sim, axis=1), marker='o', label=labels[p])
    plt.legend()
    plt.xlabel("Timesteps", fontsize=16)
    plt.ylabel("Cosine similarity", fontsize=16)
    plt.savefig(save_path + 'figure_5', dpi=400, bbox_inches='tight')
    plt.close()


# set device
device = torch.device("cuda:0")
base = '/home3/ebrahim/isr/'

if mt == 0:
    nonlin = 'sigmoid'
    model_folder = 'simulation_one_weights/'
    path = base + 'saved_models/' + model_folder + 'run_' + run_number + '/'
    with open(path + 'model_settings.pkl', 'rb') as handle:
        ms = pickle.load(handle)
    model = RNN_feedback(ms['is'], ms['hs'], ms['os'])
    save_folder = model_folder

elif mt == 1: 
    model_folder = 'simulation_one_cell/'
    path = base + 'saved_models/' + model_folder + 'run_' + run_number + '/'
    with open(path + 'model_settings.pkl', 'rb') as handle:
        ms = pickle.load(handle)
    try:
        nonlin = ms['nonlin']
    except:
        nonlin = 'sigmoid'

    model = RNN_one_layer(ms['is'], ms['hs'], ms['os'], ms['feedback_scaling'], nonlin)
    save_folder = model_folder

else:
    model_folder = 'simulation_one_cell/'
    path = base + 'saved_models/' + model_folder + 'run_' + run_number + '/'
    with open(path + 'model_settings.pkl', 'rb') as handle:
        ms = pickle.load(handle)
    nonlin = ms['nonlin']
    model = RNN_two_layers(ms['is'], ms['hs'], ms['os'], ms['feedback_scaling'], nonlin, ms['fb_type'])
    save_folder = model_folder

# specify folder for saving output
save_path = 'saved_data/' + save_folder + 'run_' + run_number + '/'
os.makedirs(save_path, exist_ok=True)

# create permutations of specified list length with a given subset of letters
num_letters = 10

if num_letters == 10:
    letters_subset = [0, 1, 4, 7, 9]
else:
    letters_subset = [0,1,4,7,10,13,16,19,22,25]

list_length = 4
rt = Regression_Trials(num_letters, letters_subset, list_length)
rt_dataloader = DataLoader(rt, batch_size=len(rt), shuffle=False)

# convert to one hot encoding, X_transform has shape num_permutations X num_features
X_reg = rt.X_reg
enc = OneHotEncoder(sparse=False)
X_transform = enc.fit_transform(X_reg.T)

model.load_state_dict(torch.load(path + 'final_model_weights.pth'))
model.to(device)

# obtain linearized model activity 
linearized_hidden_activity = regression_RNN_activity(model, rt_dataloader, len(rt), ms['loss_func'])
linearized_hidden_activity = linearized_hidden_activity.cpu().numpy()



# obtain output weights for the subset of letters
for name, param in model.named_parameters():
    if name == 'h2o.weight':
        h2o_weight = param.detach().cpu().numpy()
    if name == 'o2h.weight' or name == 'RNN.o2h.weight' or name == 'RNN2.o2h.weight':
        o2h_weight = param.detach().cpu().numpy()

h2o_weight_subset = h2o_weight[letters_subset]
plt.hist(np.ravel(h2o_weight_subset))
plt.savefig(save_path + 'h2o_weights', dpi=400, bbox_inches='tight')
plt.close()

o2h_weight_subset = o2h_weight[letters_subset]
plt.hist(np.ravel(o2h_weight_subset))
plt.savefig(save_path + 'o2h_weights', dpi=400, bbox_inches='tight')
plt.close()

MA = []
for t in [1,2,3,4,5,6,7,8]:
    MA.append([])
    r2_score_value, W = compute_element_vectors(X_transform, linearized_hidden_activity, 
    len(letters_subset), t, num_letters)
    print("t: ", t)
    print("R2 score: ", r2_score_value)
    partition_W(W, len(letters_subset), MA)
    if t == 4:
        W_encoding = W

figure_4_code(W_encoding, len(letters_subset))
figure_5_code(MA)

corr_same_avg = []
corr_diff_avg = []
X = rt.X[:, 0:8, :-1].cpu().numpy().argmax(2)
for i in range(4):
    corr_same, corr_diff = positional_similarity_simple(0,i,linearized_hidden_activity, X)
    corr_same_avg.append(np.mean(corr_same))
    corr_diff_avg.append(np.mean(corr_diff))
plt.plot(np.arange(4), corr_same_avg, marker='o', label='Same item')
plt.plot(np.arange(4), corr_diff_avg, marker='o', label='Distinct items')
plt.xlabel("Distance", fontsize=16)
plt.ylabel("Correlation", fontsize=16)
plt.xticks([0, 1, 2, 3])
plt.legend()
plt.savefig(save_path + 'pos_simple', dpi=400, bbox_inches='tight')
plt.close()



















