import torch
import numpy as np

class BasicStandardizer:
    def __init__(self):
        self.name = 'BasicStandardizer'
        self.std_gev = 10
        self.std_coord = 1
        self.idx_coord = []
        self.idx_gev = []
        self.idx_log = []

        
    def transform(self,data):
        if len(data.shape)==3:
            data[:,:,self.idx_gev]/=self.std_gev
            data[:,:,self.idx_log]=np.where((data[:,:,self.idx_log] + 1)>0, np.log(data[:,:,self.idx_log] + 1), 0)
            data[:,:,self.idx_coord]/=self.std_coord
        if len(data.shape)==2:
            data[:,self.idx_gev]/=self.std_gev
            data[:,self.idx_log]=np.where((data[:,self.idx_log] + 1)>0, np.log(data[:,self.idx_log] + 1), 0)
            data[:,self.idx_coord]/=self.std_coord
        return data

    def inverse_transform(self,data):
        if len(data.shape)==3:
            data[:,:,self.idx_log]=(np.exp(data[:,:,self.idx_log])) - 1
            data[:,:,self.idx_gev]*=self.std_gev
            data[:,:,self.idx_coord]*=self.std_coord
        if len(data.shape)==2:
            data[:,self.idx_log]=(np.exp(data[:,self.idx_log])) - 1
            data[:,self.idx_gev]*=self.std_gev
            data[:,self.idx_coord]*=self.std_coord
        return data



class Standardizer:
    def __init__(self,minmax_idx=None,  log_idx=None):
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.minmax_idx = minmax_idx
        self.std_idx = None
        self.log_idx = log_idx

    def fit(self, data):
        """
        :param data: torch tensor
        """
        num_feats = data.shape[-1]
        all_indices = np.arange(num_feats)
        if self.minmax_idx is not None :
            self.std_idx = list(set(all_indices)-set(self.minmax_idx))
            self.min = torch.min(data[:,self.minmax_idx], dim=0).values
            self.max = torch.max(data[:,self.minmax_idx], dim=0).values
        else :
            self.std_idx = all_indices
        self.mean = torch.mean(data[:,self.std_idx], dim=0)
        self.std = torch.std(data[:,self.std_idx], dim=0)



    def transform(self, data):
        if self.log_idx is not None:
            data[:,self.log_idx] = torch.log(data[:,self.log_idx] + 1)
        data[:,self.std_idx] =  (data[:,self.std_idx] - self.mean) / self.std
        if self.minmax_idx is not None :
            data[:,self.minmax_idx] =  (data[:,self.minmax_idx]-self.min)/(self.max-self.min)
        return data

    def inverse_transform(self, data):
        data[:,self.std_idx] =  (data[:,self.std_idx]* self.std) + self.mean
        if self.minmax_idx is not None :
            data[:,self.minmax_idx] =  data[:,self.minmax_idx]* (self.max-self.min) + self.min
        if self.log_idx is not None:
            data[:,self.log_idx] = (torch.exp(data[:,self.log_idx])) - 1
        return data