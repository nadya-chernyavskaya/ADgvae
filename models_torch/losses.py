import sys
import torch
import torch.nn.functional as F
import scipy.optimize
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

multi_gpu = torch.cuda.device_count()>1
eps = 1e-12
torch.autograd.set_detect_anomaly(True)


def xyze_to_ptetaphi_torch(y):
    ''' converts an array [N x 100, 4] of particles
from px, py, pz, E to pt,eta, phi
    '''
    PX, PY, PZ, E = range(4)
    pt = torch.sqrt(torch.pow(y[:,PX], 2) + torch.pow(y[:,PY], 2)) 
    eta = torch.asinh(torch.where(pt < 10e-5, torch.zeros_like(pt), torch.div(y[:,PZ], pt)))
    phi = torch.atan2(y[:,PY], y[:,PX])

    relu =  m = nn.ReLU() #inplace=True
    y_E_trimmed = relu(y[:,-1]) #trimming E
    y_pt_trimmed = relu(pt) #trimming pt
    full_y = torch.stack((y[:,0],y[:,1],y[:,2],y_E_trimmed,y_pt_trimmed,eta,phi), dim=1)

    return full_y


class LossFunction:
    def __init__(self, lossname, device=torch.device('cuda:0')):
        loss = getattr(self, lossname)
        self.name = lossname
        self.loss_ftn = loss
        self.device = device
        
    def mse(self, x, y):
        return F.mse_loss(x, y, reduction='mean')
    
    def mse_coordinates(self, y,x): #for some reason convension is : out,in
        #From px,py,pz,E get pt, eta, phi (do not predict them)
        #x is px,py,pz,E,pt,eta,phi
        #y is px,py,pz,E
        full_y = xyze_to_ptetaphi_torch(y)
        return self.mse(x,full_y)
        