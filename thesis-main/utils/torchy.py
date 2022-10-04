
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.func as func

import numpy as np

import math
from torch.utils.data import DataLoader, random_split

from sklearn.preprocessing import StandardScaler




class OT_Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, datasets):
        'Initialization'
        self.datasets = datasets

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datasets[0])

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample and label
        X = [self.datasets[0][index][0][0], self.datasets[1][index][0][1]]
        y = self.datasets[0][index][1]
        return X, y

class Pseudoset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, datasets, targets):
        'Initialization'
        self.datasets = datasets
        self.targets = targets

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.targets)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample and label
        X_training_set = self.datasets[np.random.choice([0,1])][index][0]

        
        X_pseudo_set = self.datasets[-1][index][0]

        X = [X_training_set, X_pseudo_set]


        y = self.targets[index]

        return X, y


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_list, targets):
        'Initialization'
        self.targets = targets
        self.data_list = data_list

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.targets)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample and label
        X = [d[index][0] for d in self.data_list]
        y = self.targets[index]
        return X, y


# from https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/7
class CustomNet(nn.Module):
    def __init__(self, original, scalar_means = None, scalar_stds=None):
        super().__init__()

        self.scalar_means = scalar_means
        self.scalar_stds = scalar_stds

        self.features = nn.ModuleList(original.children())[:-1]

        self.features = nn.Sequential(*self.features)
        in_features = original.fc.in_features

        self.fc0 = nn.Linear(in_features, 256)
        self.fc0_bn = nn.BatchNorm1d(256, eps = 1e-2)
        self.fc1 = nn.Linear(256, 7)
        self.fc1_bn = nn.BatchNorm1d(7, eps = 1e-2)
        
        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain = 1)

    def forward(self, input_imgs):

        output = self.features(input_imgs)

        if self.scalar_means is not None and self.scalar_stds is not None:
            output = func.feature_wise_scaling(output, self.scalar_means, self.scalar_stds)
        output = output.view(input_imgs.size(0), -1)
        embs = output

        output = self.fc0_bn(F.relu(self.fc0(output)))
        output = self.fc1_bn(F.relu(self.fc1(output)))
                
        return embs, output



def get_feature_loaders(data_dict, batch_size, train_names, valid_name=None, test_name=None, verbose=True):

    data, l = func.dict_to_data(data_dict, train_names) 
    dataset = Dataset(data, l)
    if verbose: print(f'Training on {train_names}')
    # Get validation
    if valid_name=='split':
        if verbose: print('Validation on trainingset split')
        r = 0.8
        train, val = random_split(dataset, [math.floor(r*len(dataset)), math.ceil((1-r)*len(dataset))])
    elif valid_name:
        if verbose: print(f'Validation on {valid_name}')
        train = dataset
        data, l = func.dict_to_data(data_dict, valid_name) 
        val = Dataset(data, l)
    else:
        if verbose: print('No validation')
        val =  None
        train = dataset
    
    #Get testing
    if test_name=='split':
        if verbose: print('Testing on validation set split')
        r = 0.5
        val, test = random_split(val, [math.floor(r*len(val)), math.ceil((1-r)*len(val))])
    elif test_name:
        if verbose: print(f'Testing on {test_name}')
        data, l = func.dict_to_data(data_dict, test_name) 
        test = Dataset(data, l)
    else:
        if verbose: print('No testing')
        test =  None
        train = dataset
    

    datasets = [train, val, test]

    train_loader, valid_loader, test_loader = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=0) if d else d for d in datasets]

    return train_loader, valid_loader, test_loader, datasets




def get_image_loaders(datasets, batch_size, train_names, valid_name=None, test_name=None, verbose=True, shuffle=True, ot_order=False):
    
    data, l = func.bootstrap(datasets, train_names)
    dataset = Dataset(data, l)
    if verbose: print(f'Training on {train_names}')
    # Get validation
    if valid_name=='split':
        if verbose: print('Validation on trainingset split')
        r = 0.8
        train, val = random_split(dataset, [math.floor(r*len(dataset)), math.ceil((1-r)*len(dataset))])
    elif valid_name:
        if verbose: print(f'Validation on {valid_name}')
        train = dataset
        data, l = func.bootstrap(datasets, valid_name)
        val = Dataset(data, l)
    else:
        if verbose: print('No validation')
        val =  None
        train = dataset
    
    #Get testing
    if test_name=='split':
        if verbose: print('Testing on validation set split')
        r = 0.5
        val, test = random_split(val, [math.floor(r*len(val)), math.ceil((1-r)*len(val))])
    elif test_name:
        if verbose: print(f'Testing on {test_name}')
        data, l = func.bootstrap(datasets, test_name)
        test = Dataset(data, l)
    else:
        if verbose: print('No testing')
        test =  None
        train = dataset
    
    datasets = [train, val]
    train_loader, valid_loader  = [DataLoader(d, batch_size=batch_size, shuffle=shuffle, num_workers=0) if d else d for d in datasets]
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)



    return train_loader, valid_loader, test_loader, datasets


def valid_split(dataset, r, batch_size, shuffle):
    r = 0.8
    train, val = random_split(dataset, [math.floor(r*len(dataset)), math.ceil((1-r)*len(dataset))])   
    
    return [DataLoader(d, batch_size=batch_size, shuffle=shuffle, num_workers=0) for d in [train, val]]
