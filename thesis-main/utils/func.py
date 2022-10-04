import torch

from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.utils import make_grid
import utils.plot as plot

from collections import Counter

import utils.distances as distances
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

import numpy as np
from utils.torchy import OT_Dataset
import wandb


def bootstrap(datasets, domain_names):
    data_list = []

    classes = set(datasets[domain_names[0]].targets)
    num_classes = len(set(datasets[domain_names[0]].targets))
    sizes = np.zeros((len(domain_names), num_classes))

    for i, domain_key in enumerate(domain_names):
        for j, class_size in enumerate(list(Counter(datasets[domain_key].targets).values())):
            sizes[i,j] = class_size  
    
    max_sizes = np.min(sizes, axis=0).astype(int)
    labels = torch.from_numpy(np.repeat(np.arange(num_classes), max_sizes))
    for d in domain_names:
        domain_list = []
        domain_targets = np.array(datasets[d].targets)
        for j, class_key in enumerate(classes):
            bools = domain_targets==class_key
            idx = (np.cumsum(np.ones(domain_targets.shape[0]))[bools]-1).astype(int)
            data_subset = Subset(datasets[d], idx)
            if len(data_subset) != max_sizes[j]:
                sampled_idx = np.random.choice(idx, size=max_sizes[j])
                data_subset = Subset(datasets[d], sampled_idx)
            domain_list.append(data_subset)
        data_list.append(ConcatDataset(domain_list))
        
    return data_list, labels        


def dict_to_data(feature_dict, domain_names):
    data_list = []

    num_classes = len(feature_dict[domain_names[0]])
    sizes = np.zeros((len(domain_names), num_classes))

    for i, domain_key in enumerate(domain_names):
        for j, class_key in enumerate(feature_dict[domain_key].keys()):
            sizes[i,j] =  len(feature_dict[domain_key][class_key])            
    
    max_sizes = np.max(sizes, axis=0).astype(int)
    labels = torch.from_numpy(np.repeat(np.arange(num_classes), max_sizes))
    
    for i, domain_key in enumerate(domain_names):
        domain_list = []
        for j, class_key in enumerate(feature_dict[domain_key].keys()):
            data = feature_dict[domain_key][class_key]
            data= torch.stack(data).squeeze()
            if len(data) == max_sizes[j]:
                domain_list.append(data)
            else:
                idx = np.random.randint(data.shape[0], size=max_sizes[j])
                domain_list.append(data[idx])
        
        data_list.append(torch.vstack(domain_list))
        
    return data_list, labels


def train(net, criterion, optimizer, train_loader, train_set=None, epochs=20, gamma=0.5,
            delta=1, device='cpu', valid_loader=None, 
            distance=None, embeddings=False, min_valid_loss=1000, verbose=False, 
            wb=False, model_path=None, ot_order=False):

    scalar_means, scalar_stds = 0, 0
    net.to(device)
    if verbose: 
        print(f'Using distance:{distance.__name__}')
        print(f'Cuda:{device}')

    for epoch in range(epochs):  # loop over the dataset multiple times
        
        if verbose: print(f'Epoch {epoch}')

        if ot_order==('Full' or 'Both'):
            train_loader = order_full_dataset(net, train_loader, train_set, device=device)

        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            step = epoch*len(train_loader) + i
            inputs, labels = data
            inputs, labels = inputs, labels.to(device)

            optimizer.zero_grad()

            outputs = [net(i.to(device)) for i in inputs]

            basic_loss = sum([criterion(output, labels) for _, output in outputs])
            if delta<1:
                basic_loss = criterion(outputs[0][1], labels) + delta*criterion(outputs[1][1], labels)
            if len(inputs)==2:
                if embeddings: 
                    emb1, emb2 = [emb for emb, _ in outputs]
                else:
                    emb1, emb2 = [emb for _, emb in outputs]

                if ot_order==('Batch' or 'Both'):
                    emb1, emb2, labels, _ = get_ot_order(emb1, emb2, labels)
                distance_loss = distance(emb1, emb2, device=device)
            else: 
                distance_loss = np.NaN

            if distance:
                if distance_loss == distance_loss:
                    loss = basic_loss + gamma*(distance_loss)
                    
                else:
                    if verbose: print('out of bounds, epoch/i: ', epoch,'/', i)
                    loss = basic_loss
            else:
                loss = basic_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += basic_loss.item()
            if wb:
                wandb.log({"Distance loss": distance_loss,
                "Training loss": basic_loss}, step=step)

            if verbose:
                print(f"Distance loss: {distance_loss.item()} \t Training loss: {basic_loss.item()}")

            if valid_loader:
                _, min_valid_loss, _, _  = test_model(net, valid_loader, criterion, wb, min_valid_loss, testing=False, model_path=model_path, verbose=verbose, device=device, step=step)
   
        if verbose: print(train_loss)
    if verbose: print('Finished Training')
    return min_valid_loss
    

def test_model(net, loader, criterion, wb=False, min_valid_loss=10000, 
                testing=True, model_path=None, verbose=False, device='cpu', 
                step=None, n=1, ):
    
    net.eval()
    net.to(device)
    # Testing
    if testing:
        print(f'Testing model: {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    pseudo_labels = []
    actual_labels = []

    maxes = []
    max_indices = []
    top_pseudo_indeces= []

    embeddings = []

    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        total = 0
        loss = 0
        for data, labels in loader:
            data, labels = data, labels.to(device)
            data = [d.to(device) for d in data]
            
            outputs = [net(d) for d in data]
            
            predicted = [torch.argmax(out.data, 1) for _, out in outputs]
            # print( [torch.max(out.data, 1)[0].median() for _, out in outputs[:1]])

            correct += sum([(pred == labels).sum().item() for pred in predicted])

            loss += sum([criterion(out,labels) for _, out in outputs])

            if testing:
                batch_maxes, batch_max_indices = torch.max(outputs[0][1],  1)
                maxes += batch_maxes
                max_indices += batch_max_indices
                pseudo_labels += [l.item() for l in predicted[0]]
                actual_labels += [l.item() for l in labels]

            else:
                embeddings += [out for out, _ in outputs]
            
            total_loss += loss.item() * data[0].size(0) 

            total += labels.size(0)*len(data)
            
        acc = (100 * correct / total)
        

        # Validation
        if not testing: 
            if wb:
                wandb.log({'Validation loss': total_loss/len(loader),
                'Validation accuracy': acc}, step=step)

            if min_valid_loss > total_loss:
                if verbose: print(f'Val Loss Decr({min_valid_loss:.6f}->{total_loss:.6f}) Saving {model_path}')
                min_valid_loss = total_loss
                # Saving State Dict
                if model_path:
                    torch.save(net.state_dict(), model_path) 
        
        # Testing
        else:

            if wb:
                wandb.log({'Test Loss': total_loss/len(loader),
                'Test Accuracy': acc
                }, step=step)
        


    if verbose: print(f'Accuracy of the network on the {total} test images: {acc} %')
    net.train()

    if testing: 
        top_pseudo_indeces = get_topk_indices(torch.hstack(maxes), torch.hstack(max_indices), n)

        return acc, min_valid_loss, pseudo_labels, top_pseudo_indeces.to(dtype=torch.int32, device='cpu'), actual_labels
    else:
        embeddings = torch.vstack(embeddings)
        # print(embeddings.shape)
        scaler = StandardScaler()
        scaler.fit(embeddings.cpu().numpy())

        scalar_means = scaler.mean_
        scalar_stds = scaler.scale_

        
        return acc, min_valid_loss, scalar_means, scalar_stds


def get_topk_indices(maxes, max_indices, n):

    top_pseudo_indeces = []
    for class_nr in range(torch.max(max_indices)+1):
        mask = (max_indices==class_nr)
        count = torch.count_nonzero(mask)
        k =  int(n*count)
        
        _, class_top_indeces = torch.topk(maxes*mask, k, sorted=False)
        top_pseudo_indeces += class_top_indeces

    return torch.hstack(top_pseudo_indeces).sort().values



def get_ot_order(emb1, emb2, labels):
    
    emb1_list = []
    emb2_list = []
    label_list = []
    indeces = {}

    for i in range(7):
        bools = labels==i

        
        emb1_subset, emb2_subset = emb1[bools], emb2[bools]
        M = distances.optimal_transport(emb1_subset, emb2_subset)
        _, col_ind = linear_sum_assignment(M.to('cpu').detach().numpy())

        indeces[i] = col_ind

        # emb1_subset = emb1_subset[col_ind]
        emb2_subset = emb2_subset[col_ind]

        emb1_list.append(emb1_subset)
        emb2_list.append(emb2_subset)
        label_list.append(labels[bools])

    return torch.vstack(emb1_list), torch.vstack(emb2_list), torch.hstack(label_list), indeces


def order_full_dataset(net, dataloader, train_set, device='cpu', model_path=None, batch_size=128, shuffle=True, num_workers=0):

    if model_path:
        print(f'Loading model: {model_path}')
        net.load_state_dict(torch.load(model_path))

    big_emb1 = []
    big_emb2 = []
    label_list = []
    print('Getting embeddings...')
    with torch.no_grad():

        for data, labels in dataloader:

            emb1, emb2 = [net(d.to(device))[0] for d in data]

            big_emb1.append(emb1.detach().cpu())
            big_emb2.append(emb2.detach().cpu())
            label_list.append(labels)

    print('Getting ot order indeces...')
    _, _, _, indeces = get_ot_order(torch.vstack(big_emb1), torch.vstack(big_emb2), torch.hstack(label_list))

    domain_targets = np.array(train_set.labels)
    dataset_list = []
    for j in range(max(domain_targets)+1):
        bools = domain_targets==j

        idx = (np.cumsum(np.ones(domain_targets.shape[0]))[bools]-1).astype(int)
        domain1_data_subset = Subset(train_set, idx)
        domain2_data_subset = Subset(Subset(train_set, idx), indeces[j])

        dataset_list.append(OT_Dataset((domain1_data_subset, domain2_data_subset)))
            
    return DataLoader(ConcatDataset(dataset_list), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_pseudo_labels(net, test_loader, device):
    
    net.eval()
    net.to(device)

    pseudo_labels = []

    with torch.no_grad():

        for data, labels in test_loader:
            data, labels = data, labels.to(device)
            data = [d.to(device) for d in data]

            outputs = [net(d) for d in data]
            predicted = [torch.max(out.data, 1).indices for _, out in outputs]
            pseudo_labels+= predicted[0]

    return pseudo_labels 

def feature_wise_scaling(feature_batch: torch.Tensor, scalar_means: torch.Tensor, scalar_stds: torch.Tensor):
    device = feature_batch.device
    feature_batch = feature_batch.detach().cpu().numpy().squeeze()

    # Scale to mean=0 std=1
    scaler = StandardScaler()
    scaled_feature_batch = scaler.fit_transform(feature_batch)

    # Scale back to training data means and stds
    scaler.mean_ = scalar_means
    scaler.scale_ = scalar_stds
    scaled_feature_batch = scaler.inverse_transform(scaled_feature_batch)
    # scaled_feature_batch += scalar_means
    # scaled_feature_batch *= scalar_stds

    return torch.from_numpy(scaled_feature_batch).to(device)



   

def check_errors(net, loader, class_names, model_paths=None, save=True, device='cpu'):
    
    net.eval()
    net.to(device)
    # Testing

    maxes = {}
    max_indices = {}
    pseudo_labels = {}
    actual_labels = {}
    examples = {'label':[], 'data' : [], 'confidence':[], 'actualLabels':[]}
    actual_labels = []
    all_data = []
    for i, model_path in enumerate(model_paths):

        print(f'Testing model for error printing: {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

        maxes[i] = []
        max_indices[i] = []
        pseudo_labels[i] = []


        with torch.no_grad():

            for data, labels in loader:
                data, labels = data, labels.to(device)
                data = [d.to(device) for d in data]
                
                outputs = [net(d) for d in data]
                
                predicted = [torch.argmax(out.data, 1) for _, out in outputs]
                # print( [torch.max(out.data, 1)[0].median() for _, out in outputs[:1]])
                batch_maxes, batch_max_indices = torch.max(outputs[0][1],  1)

                maxes[i] += batch_maxes
                max_indices[i] += batch_max_indices
                pseudo_labels[i] += [l.item() for l in predicted[0]]

                if i ==0: 
                    all_data += [d for d in data]
                    actual_labels += [l.item() for l in labels]

            #i=0 is baseline
            if i >0:
                all_data = torch.vstack(all_data)
                for vice_versa in [False, True]:
                    for j, prediction in enumerate(pseudo_labels[i]):

                        if not vice_versa:
                            v = ''
                            condition = (prediction != actual_labels[j]) and (pseudo_labels[0][j] == actual_labels[j])
                        else:
                            v = 'vice'
                            condition =  (prediction == actual_labels[j]) and (pseudo_labels[0][j] != actual_labels[j])

                        if condition:
                            examples['actualLabels'].append(actual_labels[j])
                            examples['label'].append(prediction)
                            examples['data'].append(all_data[j])
                            examples['confidence'].append(maxes[i][j])

                    if save:
                        inputs = []
                        predicted_classes = []
                        confidences = []
                        for class_, class_name in enumerate(class_names):
                            print(class_name)
                            class_mask = (torch.Tensor(examples['actualLabels'])==class_)
                            _, class_top_indeces = torch.topk(torch.Tensor(examples['confidence'])*class_mask, 1, sorted=True)
                            
                            inputs.append(torch.stack(examples['data'])[class_top_indeces])
                            predicted_classes.append(np.array(examples['label'])[class_top_indeces])
                            confidences.append(torch.vstack(examples['confidence'])[class_top_indeces])
                        print(torch.vstack(inputs).shape)
                        out = make_grid(torch.vstack(inputs)).cpu()
                        plot.imshow(out, title=[class_names[c] for c in predicted_classes], model_path=model_path, class_name=v)

