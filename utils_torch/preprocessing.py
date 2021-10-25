import torch
from utils_torch.scaler import Standardizer

def standardize(train_dataset, valid_dataset=None, test_dataset=None,minmax_idx = None, log_idx=None):
    """
    standardize dataset and return scaler for inversion
    :param train_dataset: list of Data objects
    :param valid_dataset: list of Data objects
    :param test_dataset: list of Data objects
    :param log_pt: log pt before standardization
    :return scaler: sklearn StandardScaler
    """
    train_x = torch.cat([d.x for d in train_dataset])
    if log_idx is not None:
        train_x[:,log_idx] = torch.log(train_x[:,log_idx] + 1)

    scaler = Standardizer(minmax_idx=minmax_idx, log_idx=log_idx)
    scaler.fit(train_x)
    if (valid_dataset is not None) and (test_dataset is not None) :
        datasets = (train_dataset, valid_dataset, test_dataset)
    elif (valid_dataset is not None) and (test_dataset is None) :
        datasets = (train_dataset, valid_dataset)
    else:
        datasets = [train_dataset]
    for dataset in datasets:
        for d in dataset:
            d.x[:,:] = scaler.transform(d.x)
    return scaler


def basic_standarize(train_dataset,log_idx=None):
