import torch
from utils_torch.scaler import Standardizer

def standardize(train_dataset, valid_dataset=None, test_dataset=None, log_pt=False):
    """
    standardize dataset and return scaler for inversion
    :param train_dataset: list of Data objects
    :param valid_dataset: list of Data objects
    :param test_dataset: list of Data objects
    :param log_pt: log pt before standardization
    :return scaler: sklearn StandardScaler
    """
    train_x = torch.cat([d.x for d in train_dataset])
    if log_pt:
        train_x[:,0] = torch.log(train_x[:,0] + 1)

    scaler = Standardizer()
    scaler.fit(train_x)
    if (valid_dataset is not None) and (test_dataset is not None) :
        datasets = (train_dataset, valid_dataset, test_dataset)
    else:
        datasets = [train_dataset]
    for dataset in datasets:
        for d in dataset:
            d.x[:,:] = scaler.transform(d.x)
    return scaler
