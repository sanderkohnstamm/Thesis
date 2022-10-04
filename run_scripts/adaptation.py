from __future__ import print_function, division
from dis import dis
from distutils.errors import DistutilsInternalError
from random import shuffle
from tabnanny import verbose
import copy

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset

from torchvision import models, transforms, datasets
import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.

cudnn.benchmark = True

import utils.func as func
import utils.torchy as torchy
import utils.distances as distances
import utils.plot as plot

from collections import Counter
from sklearn.metrics import confusion_matrix

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

    criterion = nn.CrossEntropyLoss()

    train_names = [ 'photo', 'art_painting']
    valid_name = 'split'
    test_name = ['cartoon']

    lr=0.001
    adapt_lrs = [lr, lr/2, lr/4, lr/8, lr/10]
    adapt_lr = adapt_lrs[-1]
    distances_ = [distances.hsic_normalized, distances.optimal_transport_loss, distances.norm]
    batch_size = 128
    epochs = 10
    verbose = True
    now = datetime.now()
    embeddings = True
    runs = 2
    pretrained = True

    wandb_ = True
    tsne = True
    cf = False
  
    gamma = 0
    distance = distances_[0]
    ot_order = False
    n_pseudo_split = 0.5
    delta=1
    scaling = False
    ot_orders = [ False, 'Full', 'Both', 'Batch']

    for i in range(runs):

        train_loader, valid_loader, test_loader, new_datasets  = torchy.get_image_loaders(ds, 
                                                                    batch_size,
                                                                    train_names,
                                                                    valid_name,
                                                                    test_name,
                                                                    verbose=verbose
                                                                    )

        full_loader, _, _, _  = torchy.get_image_loaders(ds, 
                                                    batch_size,
                                                    train_names=test_name,
                                                    verbose=verbose
                                                    )
        
        current_time = now.strftime("%H_%M_%S")
        model_name =f'{current_time}_{i}'
        model_path = models_path + model_name + '.pth'

        resnet18 = models.resnet18(pretrained=pretrained)
        net = torchy.CustomNet(resnet18)


        min_valid_loss = 10000

        ds_adapt = copy.deepcopy(ds)

        optimizer = optim.Adam(net.parameters(), lr=lr)
        min_valid_loss = func.train(net, criterion, optimizer, 
                                    train_loader, train_set=new_datasets[0],
                                    valid_loader=valid_loader,
                                    epochs=10,
                                    distance=distance,
                                    gamma=0,
                                    delta=1,
                                    device=device,
                                    embeddings=embeddings,
                                    min_valid_loss = min_valid_loss,
                                    wb=False,
                                    model_path=model_path,
                                    verbose=verbose,
                                    ot_order=ot_order,
                                    )

                
        #test model
        acc, min_valid_loss, pseudo_labels, top_pseudo_indeces, actual_labels = func.test_model(net, test_loader, criterion,
                                    wb=False, 
                                    min_valid_loss=min_valid_loss, 
                                    testing=True, 
                                    model_path=model_path, 
                                    verbose=verbose, 
                                    device=device,
                                    n=n_pseudo_split
                                    )
        if scaling:
            _, _, scalar_means, scalar_stds = func.test_model(net, train_loader, criterion,
                                        wb=False, 
                                        min_valid_loss=min_valid_loss, 
                                        testing=False, 
                                        model_path=model_path, 
                                        verbose=verbose, 
                                        device=device,
                                        n=n_pseudo_split
                                        )
        else:
            scalar_means, scalar_stds = None, None
        
        correct = []
        # print(len(ds_adapt[test_name[0]]))
        for idx in range(len(pseudo_labels)):
            if ds_adapt[test_name[0]].targets[idx]==pseudo_labels[idx]:
                correct.append(pseudo_labels[idx])

        new_pseudo_labels = []
        for j in top_pseudo_indeces:
            new_pseudo_labels.append(pseudo_labels[j])

        ds_adapt[test_name[0]] = Subset(ds_adapt[test_name[0]], top_pseudo_indeces)
        ds_adapt[test_name[0]].targets = new_pseudo_labels


        print('start: ', acc, len(correct)/len(pseudo_labels), Counter(correct))

        

        data, l = func.bootstrap(ds_adapt, train_names+test_name)

        adaptation_set = torchy.Pseudoset(data, l)
        
        train_loader_adapt, valid_loader_adapt = torchy.valid_split(adaptation_set, r=0.8, batch_size=batch_size, shuffle=shuffle)
        
        if tsne:
            plot.make_tsne(net, model_path, full_loader, domain_names, classes_names, test_name=test_name, wb=False)

        if cf:
            labels = (actual_labels, pseudo_labels)
            plot.make_confusion_matrix(labels, categories=classes_names, model_path=model_path, verbose=verbose)     
        for gamma in [0]:
            for distance in distances_[:1]:
                for n_pseudo_split in [0.5]:
                    for adapt_lr in [lr/2]:
                        adaptation_model_path = models_path +'adapt'+ model_name +  f'{gamma}'.replace(".", "")+ str(distance.__name__)+'.pth'
                        for delta in [1]:
                            if wandb_:
                                wandb.init(project=f"tno_adaptation_cartoon",
                                            entity="skohnie",
                                            name=f'{current_time}/{gamma}/{adapt_lr}/{str(distance.__name__)}/{i}',
                                            config = {"learning_rate": adapt_lr,
                                                        "epochs": epochs,
                                                        "batch_size": batch_size,
                                                        "gamma": gamma,
                                                        "Train names": train_names,
                                                        "Valid name": valid_name,
                                                        "Test name": test_name,
                                                        "Use embeddings":embeddings,
                                                        "Pretrained": pretrained,
                                                        "Use distance": distance.__name__,
                                                        "Ordering with OT": ot_order,
                                                        "Base Accuracy": acc,
                                                        "Model name": model_name,
                                                        "Pseudo split": n_pseudo_split,
                                                        "Delta": delta,
                                                        "Scaling": scaling,
                                                        }
                                        )

                            net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
                            net.scalar_means, net.scalar_stds = scalar_means, scalar_stds
                            optimizer = optim.Adam(net.parameters(), lr=adapt_lr)
                            min_valid_loss = func.train(net, criterion, optimizer, 
                                                        train_loader_adapt, train_set=new_datasets[0],
                                                        valid_loader=valid_loader_adapt,
                                                        epochs=epochs,
                                                        distance=distance,
                                                        gamma=gamma,
                                                        delta=delta,
                                                        device=device,
                                                        embeddings=embeddings,
                                                        min_valid_loss = 100000,
                                                        wb=wandb_,
                                                        model_path=adaptation_model_path,
                                                        verbose=verbose,
                                                        ot_order=ot_order,
                                                        )


                            acc_new ,_, pseudo_labels,_, actual_labels = func.test_model(net, test_loader, criterion,
                                                        wb=False, 
                                                        min_valid_loss=min_valid_loss, 
                                                        testing=True, 
                                                        model_path=adaptation_model_path, 
                                                        verbose=verbose, 
                                                        device=device,
                                                        )
                            print(gamma, ':', acc_new, '-', acc, '|', adaptation_model_path)
                            
                            if wandb_:
                                wandb.log({'Acc difference': acc_new-acc,
                                'Acc begin': acc
                                })

                            if tsne:
                                plot.make_tsne(net, adaptation_model_path, full_loader, domain_names, classes_names, test_name=test_name, wb=False)
                            if cf:
                                labels = (actual_labels, pseudo_labels)
                                plot.make_confusion_matrix(labels, categories=classes_names, model_path=adaptation_model_path, verbose=verbose)  
                            if wandb_: wandb.finish()

                                                                                

                                


                        
                                
