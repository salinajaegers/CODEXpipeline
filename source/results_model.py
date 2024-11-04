##################################################################################################
# Inspect output of the classifier: confusion table, accuracy... Plot tops classification sample #
##################################################################################################

import torch
import numpy as np
import seaborn as sns
from torch.utils.data import DataLoader
from class_dataset import myDataset, ToTensor, Subtract, RandomCrop
from load_data import DataProcesser
from torchvision import transforms
import pandas as pd
import os
import re
from utils import model_output
import matplotlib 
matplotlib.use('Agg')


def confusion_matrix(model, dataloader, device=None, labels_classes=None):
    """
    Return confusion matrix for all element in dataloader.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param labels_classes: a dictionary or pandas.Series to rename classes indexes.
    :return: A pandas DataFrame, actual classes in rows, predicted classes in columns.
    """
    from pandas.api.types import CategoricalDtype
    df_out = model_output(model, dataloader, export_prob=True, export_feat=False, device=device)
    prob_cols = [col for col in df_out.columns if col.startswith('Prob_')]
    df_out['Prediction_colname'] = df_out[prob_cols].idxmax(axis=1)  # returns name of columns
    df_out['Prediction'] = df_out['Prediction_colname'].str.replace('^Prob_', '').astype('int')
    cat_type = CategoricalDtype(categories=sorted(df_out['Class'].unique()), ordered=True)
    confmat = pd.crosstab(df_out['Class'].astype(cat_type), df_out['Prediction'].astype(cat_type), dropna=False)
    if labels_classes is not None:
        confmat.rename(labels_classes, axis='index', inplace=True)
        confmat.rename(labels_classes, axis='columns', inplace=True)
    return confmat


def acc_per_class(model, dataloader, device=None, labels_classes=None):
    """
    Return accuracy of classification per class.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param labels_classes: a dictionary or pandas.Series to rename classes indexes.
    :return: A pandas Series with accuracies per class, index contains the class names.
    """
    confmat = confusion_matrix(model, dataloader, device)
    tot_count = confmat.sum(axis=1)
    diag_count = confmat.values[[np.arange(confmat.shape[0])] * 2]
    acc = diag_count / tot_count
    if labels_classes is not None:
        acc.rename(labels_classes, axis='index', inplace=True)
    return acc


def top_confidence_perclass(model, dataloader, n=10, mode ='highest', device=None, softmax=True, labels_classes=None):
    """
    Returns the results of classification with highest or lowest confidence per class.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param n: int, the number of trajectories to return per class.
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param softmax: bool, whether to apply softmax to before selecting th trajectories.
    :param mode: str, one of ['highest', 'lowest'].
    :param labels_classes: a dictionary or pandas.Series to rename classes indexes.
    :return: A pandas DataFrame with columns: 'ID', 'Class', 'Prob_XXX' where XXX is the class index.
    """
    assert mode in ['highest', 'lowest']
    out = []
    df_out = model_output(model, dataloader, export_prob=True, export_feat=False, softmax=softmax, device=device)
    for iclass in range(len((df_out['Class'].unique()))):
        sort_by = 'Prob_' + str(iclass)
        if mode == 'highest':
            out.append(df_out.loc[df_out['Class']==iclass].sort_values(by=sort_by).tail(n))
        elif mode == 'lowest':
            out.append(df_out.loc[df_out['Class']==iclass].sort_values(by=sort_by).head(n))
    out = pd.concat(out, axis=0)
    print(out)
    if labels_classes is not None:
        out['Class'].replace(labels_classes, inplace=True)
        old_labels = {col: re.search('\d+$', col) for col in out.columns.values}
        print(old_labels)
        print(labels_classes)
        new_labels = {col: re.sub('\d+$', labels_classes[int(old_labels[col].group())], col)
                      for col in old_labels if old_labels[col]}
        out.rename(new_labels, axis='columns', inplace=True)
    return out


def worst_classification_perclass(model, dataloader, n=10, device=None, softmax=True, labels_classes=None):
    """
    Returns the worst classification per class. Worst classifications are defined as incorrect classification (i.e. the
    model predicted a class that is not the one of individual) with largest confidence.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param n: int, the maximum number of trajectories to return per class.
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param softmax: bool, whether to apply softmax to before selecting th trajectories.
    :param labels_classes: a dictionary or pandas.Series to rename classes indexes.
    :return: A pandas DataFrame with columns: 'ID', 'Class', 'Prob_XXX' where XXX is the class index.
    """
    out = []
    df_out = model_output(model, dataloader, export_prob=True, export_feat=False, softmax=softmax, device=device)
    prob_cols = [col for col in df_out.columns if col.startswith('Prob_')]
    df_out['Prediction_colname'] = df_out[prob_cols].idxmax(axis=1)  # returns name of columns
    df_out['Prediction'] = df_out['Prediction_colname'].str.replace('^Prob_', '').astype('int')
    df_out = df_out.reindex(columns=['ID', 'Class', 'Prediction', 'Prediction_colname'] + prob_cols)
    for classe in df_out['Class'].unique():
        # Cases where real class is different from the predicted one but where confidence is high for the predicted
        to_append = df_out.loc[(df_out['Class'] != df_out['Prediction']) &
                               (df_out['Class'] == classe)].copy()
        # Skip if no wrong classification for this class
        if to_append.shape[0] == 0:
            print('No wrong classificattion for class: {}'.format(classe))
            continue
        # Report value of predicted class on each row
        to_append['Prediction_confidence'] = to_append.lookup(to_append.index, to_append.Prediction_colname)
        to_append.sort_values(by='Prediction_confidence', inplace=True)
        to_append = to_append.tail(n)
        out.append(to_append)
    out = pd.concat(out, axis=0).drop(columns=['Prediction_colname', 'Prediction_confidence'])
    if labels_classes is not None:
        out['Class'].replace(labels_classes, inplace=True)
        out['Prediction'].replace(labels_classes, inplace=True)
        old_labels = {col: re.search('\d+$', col) for col in out.columns.values}
        new_labels = {col: re.sub('\d+$', labels_classes[int(old_labels[col].group())], col)
                      for col in old_labels if old_labels[col]}
        out.rename(new_labels, axis='columns', inplace=True)
    return out


def least_correlated_set(model, dataloader, threshold_confidence=0.5, n=10, init_set='medoid', corr='pearson',
                         feature_layer='pool', device=None, labels_classes=None, seed=7):
    """
    Return a set of least correlated trajectories for each class in a feature space returned by CNN.
    The procedure relies on a greedy algorithm that grows sets by adding least correlated trajectory until required
    number of trajectories is reached. A first step of trajectories selects trajectories that were correctly classified,
    optionally with a minimal level of confidence.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param threshold_confidence: float, in pre-selection step, minimal confidence of trajectory classification.
    Trajectories that don't reach this threshold won't be considered.
    :param n: int, number of trajectories in least correlated sets.
    :param init_set: str, one of ['medoid', 'centroid', 'random']. Method to choose the first elements of sets before
    greedy extension. 'medoid' initializes sets with the trajectory that minimizes the distance to all other
    trajectories. 'centroid' is the same but uses mean instead of median. 'random' chooses a random start.
    :param corr: str, correlation method. Must be one of {‘pearson’, ‘kendall’, ‘spearman’}, see
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html.
    :param feature_layer: str, name of the model module from which to hook output. See model_output().
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param labels_classes: a dictionary or pandas.Series to rename classes indexes.
    :param seed: int, seed used to determine random set initialization if init_set is 'random'.
    :return: a pandas DataFrame with least correlated sets and columns: ID, Class,
    'Prob_XXX' where XXX is the class name,  'Feat_I' where I is an increasing integer starting at 0 for each element in
    the hooked layer output.
    """
    assert init_set in ['medoid', 'centroid', 'random']
    out = []
    df_out = model_output(model, dataloader, export_prob=True, export_feat=True, device=device,
                          feature_layer=feature_layer, softmax=True)
    prob_cols = [col for col in df_out.columns if col.startswith('Prob_')]
    feat_cols = [col for col in df_out.columns if col.startswith('Feat_')]
    for iclass in range(len((df_out['Class'].unique()))):
        prob_col = 'Prob_' + str(iclass)
        df_sel = df_out.loc[(df_out['Class'] == iclass) & (df_out[prob_col] >= threshold_confidence)]
        df_sel = df_sel[['ID'] + feat_cols]
        if df_sel.shape[0] == 0:
            print('No individual reaching "threshold_confidence" for class: {}'.format(iclass))
            continue
        # --------------------------------------------------------
        # Greedy set selection of uncorrelated examples
        set_class = []
        # 0) Get correlation matrix for this class
        df_sel.set_index('ID', drop=True, inplace=True)
        df_sel = df_sel.transpose()
        df_corr = df_sel.corr(method=corr)
        # 1) Choose medoid (highest median corr)
        if init_set == 'medoid':
            set_class.append(df_corr.median(axis=0).idxmax())
        elif init_set == 'centroid':
            set_class.append(df_corr.mean(axis=0).idxmax())
        elif init_set == 'random':
            set_class.append(df_corr.sample(1, random_state=seed).index[0])
        # 2) Add least correlated series to the set
        for k in range(1, n):
            temp = df_corr.loc[set_class].drop(set_class, axis=1)
            if temp.shape[1] == 0:
                print('Exhausted class: {} after {} samples.'.format(iclass, k))
                break
            set_class.append(temp.median(axis=0).idxmin())
        # --------------------------------------------------------

        out.append(df_out[df_out['ID'].isin(set_class)])
    out = pd.concat(out, axis=0)
    if labels_classes is not None:
        out['Class'].replace(labels_classes, inplace=True)
        old_labels = {col: re.search('\d+$', col) for col in prob_cols}
        new_labels = {col: re.sub('\d+$', labels_classes[int(old_labels[col].group())], col)
                      for col in old_labels if old_labels[col]}
        out.rename(new_labels, axis='columns', inplace=True)
    return out