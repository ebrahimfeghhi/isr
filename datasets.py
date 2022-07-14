import torch 
from torch.utils.data import Dataset
import numpy as np

class OneHotLetters(Dataset):
  
    def __init__(self, max_length, num_cycles, num_letters=26, test_mode=False,
    num_test_trials=5000):

        """ Initialize class to generate letters, represented as one hot vectors in 26 dimensional space. 
        @param max_length: maximum number of letters 
        @param num_cycles: number of cycles (1 cycle = set of lists of length 1,...,max_length)
        @param num_letters: number of letters in vocabulary 
        @param test_mode: if in test_mode, only generate lists of a specified length
        @param num_test_trials: number of trials in test set
        @param seeds: A list of seeds equal to the length of the dataset.
        If specified, trials will be generated in a reproducible manner. 
        """ 

        self.max_length = max_length
        self.num_letters = num_letters
        self.num_cycles = num_cycles
        self.test_mode = test_mode
        self.storage = []
        self.num_test_trials = num_test_trials
    
    def __len__(self):

        if self.test_mode:
            return self.num_test_trials
        else: 
            return self.num_cycles * self.max_length

    def __getitem__(self, idx):

        '''
        Generates a training example
        '''

        if self.test_mode:
            rng = np.random.default_rng(idx)
            list_length = self.max_length
        else:
            rng = np.random.default_rng()
            list_length = (idx % self.max_length) + 1

        # letters (selected from 0-25)
        # recall cue (26) for remainder of input
        letters = rng.choice(self.num_letters, list_length, replace=False) 
        recall_cue = np.ones(list_length+1) * self.num_letters 
        X = torch.nn.functional.one_hot(torch.from_numpy(np.hstack((letters, recall_cue))).to(torch.long),
        num_classes=self.num_letters+1)
        
        # output is letters during letter presentation
        # letters again after recall cue
        # and finally end of list cue 
        y = torch.from_numpy(np.hstack((letters, letters, self.num_letters)))
        y = torch.nn.functional.one_hot(y, num_classes=self.num_letters+1)
        return X.to(torch.float32), y.to(torch.float32)
