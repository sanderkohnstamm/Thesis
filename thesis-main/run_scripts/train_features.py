from __future__ import print_function, division

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


cudnn.benchmark = True

import pickle
import utils.func as func
import utils.torchy as torchy

import wandb


if __name__=='__main__':
    a_file = open("../../Data/PACS.pkl", "rb")
    full_dict = pickle.load(a_file)

    all_domain_names = ['photo', 'art_painting', 'cartoon', 'sketch']
    domain_names = ['photo', 'art_painting', 'cartoon']
        
    num_classes = 7      # 7 classes for each domain: 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'
    classes_names = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']
    domain_mapping = {'photo':0, 'art_painting':1, 'cartoon':2, 'sketch':3}
    model_path = 'saved_model.pth'

    criterion = nn.CrossEntropyLoss()



    train_names = ['photo', 'cartoon']
    valid_name = ['art_painting']
    test_name = 'split'

    net = torchy.Net()
    lr=0.001
    use_hsic = True
    batch_size = 128
    epochs = 30
    for gamma in [2.5]:
        print('Lambda: ', gamma)
        for i in range(20):
            wandb.init(project="test-one",
                        entity="skohnie",
                        name=f'{gamma}/{i}',
                        config = {"learning_rate": lr,
                                    "epochs": 100,
                                    "batch_size": 128,
                                    "gamma": gamma,
                                    "Train names": train_names,
                                    "Valid name": valid_name,
                                    "Test name": test_name
                                    }
                    )



            train_loader, valid_loader, test_loader  = torchy.get_feature_loaders(full_dict, 
                                                                                batch_size,
                                                                                train_names,
                                                                                valid_name,
                                                                                test_name,
                                                                                verbose=False)


            net = torchy.Net()

            min_valid_loss = 1000



            optimizer = optim.Adam(net.parameters(), lr=lr)
            min_valid_loss = func.train(net, criterion, optimizer, 
                                        train_loader,
                                        valid_loader=valid_loader,
                                        epochs=30,
                                        use_hsic=use_hsic,
                                        gamma=gamma,
                                        writer=None,
                                        min_valid_loss = min_valid_loss,
                                        wb=True)

            acc = func.test_model(test_loader, 'saved_model.pth')
            wandb.summary['Test Accuracy'] = acc
            wandb.finish()
