from __future__ import print_function, division
from tabnanny import verbose

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


from torchvision import models, transforms, datasets
import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.


cudnn.benchmark = True

import numpy as np
import utils.func as func
import utils.torchy as torchy
import utils.distances as distances
import utils.plot as plot

from torch.utils.data import DataLoader, random_split

import wandb

# Log in to your W&B account
wandb.login()

from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter

import gc


if __name__=='__main__':
    data_root = "../data/Homework3-PACS/PACS/"
    gc.collect()
    torch.cuda.empty_cache()

    #None if no save
    models_path = '../models/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_domain_names = ['photo', 'art_painting', 'cartoon', 'sketch']
    domain_names = ['photo', 'art_painting', 'cartoon']

    # means and standard deviations ImageNet because the network is pretrained
    means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # Define transforms to apply to each image
    transf = transforms.Compose([ #transforms.Resize(227),      # Resizes short size of the PIL image to 256
                                transforms.CenterCrop(224),  # Crops a central square patch of the image 224 because torchvision's AlexNet needs a 224x224 input!
                                transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
    ])

    ds = {}
    for name in os.listdir(data_root):
    
        if not name[0] == '.':
            dataset = datasets.ImageFolder(data_root+name, transform=transf)

            ds[name] = dataset
            print(f"Added :{name}, length: {len(ds[name])}")
    
    # print(dict(Counter(dataset.targets)))
    num_classes = 7      # 7 classes for each domain: 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'
    classes_names = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']


    train_names = [ 'photo', 'cartoon']
    valid_name = ['art_painting']
    test_name = 'split'
    batch_size = 128


    train_loader, valid_loader, test_loader, ds  = torchy.get_image_loaders(ds, 
                                                                        batch_size,
                                                                        train_names,
                                                                        valid_name,
                                                                        test_name,
                                                                        verbose=verbose,
                                                                        shuffle=False)

    resnet18 = models.resnet18(pretrained=True)
    net = torchy.CustomNet(resnet18)

    indeces = func.order_full_dataset(net, train_loader, train_names)
    trainset = func.get_ordered_data(ds[0], indeces)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(train_loader)