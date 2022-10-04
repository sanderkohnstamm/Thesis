import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import wandb

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


def make_tsne(net, model_path, loader, domain_names, classes_names, test_name, wb=False, verbose=False):

    if verbose: print(f'Making tsne for: {model_path}')

    net.load_state_dict(torch.load(model_path))

    net.eval()
    net.to(torch.device('cpu'))
    test_targets = []
    test_domains = []
    test_embeddings = torch.zeros((0, 512), dtype=torch.float32)
    for X,y in loader:
        for d, x in enumerate(X):
            embeddings, logits = net(x)
            test_targets.extend(y.detach().cpu().tolist())
            test_domains+=[d]*len(x)
            test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)
    if verbose: print('Embeddings done')
    test_targets = np.array(test_targets)
    test_domains = np.array(test_domains)

    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    num_categories = len(classes_names)

    # #{'photo':0, 'art_painting':1, 'cartoon':2, 'sketch':3}
    domain_names = [test_name]
    fig, axs = plt.subplots(1, 1)
    # fig.suptitle(f'Test name {test_name}', fontsize=16)
    for i, d_n in enumerate(domain_names):
        ax = axs
        ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        # ax.set_title(d_n)
        # if d_n != test_name:
        #     continue
        for lab in range(num_categories):        
            ax.scatter(tsne_proj[(test_targets==lab) & (test_domains==i), 0],
                       tsne_proj[(test_targets==lab) & (test_domains==i),1],
                       c=np.array(cmap(lab)).reshape(1,4),
                       label = classes_names[lab],
                       alpha=0.5)
        
    # ax.legend(fontsize='small', markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5))



    if wb:
        wandb.Image(plt)
        wandb.log({model_path: plt})
        if verbose: print("Saved to wandb")

    path = model_path.split('/')[-1].split('.')[0]
    plt.savefig(f"../figs/tsne/{path}.pdf", dpi=150)
    if verbose: print(f"Saved to ../figs/tsne/{path}.pdf")

    return None


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def imshow(inp, title=None, model_path=None, class_name=''):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    path = model_path.split('/')[-1].split('.')[0]
    plt.savefig(f"../figs/error/{path}_{class_name}.pdf", dpi=150)
    plt.pause(0.001)  # pause a bit so that plots are updated





def make_confusion_matrix(labels,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=False,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          model_path=None,
                          verbose=False):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''

    actual_labels, pseudo_labels = labels
    cf = confusion_matrix(actual_labels, pseudo_labels)
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

    path = model_path.split('/')[-1].split('.')[0]
    plt.savefig(f"../figs/cf/{path}.pdf", dpi=150)
    if verbose: print(f"Saved to ../figs/{path}.pdf")

    return None