import sys
import torch
import torch.nn.functional as F
import scipy.optimize
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch.autograd import Variable
import math

multi_gpu = torch.cuda.device_count()>1
eps = 1e-12
torch.autograd.set_detect_anomaly(True)


# N(x | mu, var) = 1/sqrt{2pi var} exp[-1/(2 var) (x-mean)(x-mean)]
# log N(x| mu, var) = -log sqrt(2pi) -0.5 log var - 0.5 (x-mean)(x-mean)/var


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm



def kl_loss_manual(z_mean,z_log_var):
    kl = 1. + z_log_var - np.square(z_mean) - np.exp(z_log_var)
    kl = -0.5 * np.mean(kl, axis=-1)
    return np.array(kl)


def mse_manual(self, y, x):#for some reason convension is : out,in
    PX_idx, PY_idx, PZ_idx, E_idx, PT_idx, ETA_idx, PHI_idx = range(7)
    y_phi = math.pi*np.tanh(y[:,PHI_idx])
    y_eta = 2.5*np.tanh(y[:,ETA_idx]) #probably should increase this
    full_y = np.stack((y[:,PX_idx],y[:,PY_idx],y[:,PZ_idx],y[:,E_idx],y[:,PT_idx],y_eta,y_phi), axis=1)
    mse_loss = np.mean( np.square(x-y), axis=-1)
    return np.array(mse_loss)



def xyze_to_ptetaphi_torch(y,log_idx=[]):
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).type(torch.float32)
    PX_idx, PY_idx, PZ_idx, E_idx = range(4)
    pt =torch.sqrt(torch.pow(y[:,PX_idx], 2) + torch.pow(y[:,PY_idx], 2)) 
    eta = torch.asinh(torch.where(pt < 10e-5, torch.zeros_like(pt), torch.div(y[:,PZ_idx], pt)))
    phi = torch.atan2(y[:,PY_idx], y[:,PX_idx])
    #recalculate E as well
    #E = torch.sqrt(torch.pow(pt,2) + torch.pow(y[:,PZ_idx], 2))
    #y_out = [y[:,PX_idx],y[:,PY_idx],y[:,PZ_idx],E,pt,eta,phi]
    y_out = [y[:,PX_idx],y[:,PY_idx],y[:,PZ_idx],y[:,E_idx],pt,eta,phi]
    if len(log_idx)!=0:
      #  for idx in log_idx:
      #      y_out[idx] = torch.where((y_out[idx] + 1)>0, torch.log(y_out[idx] + 1),torch.zeros_like(y_out[idx]))
        y_out[E_idx+1] = torch.where((y_out[E_idx+1] + 1)>0, torch.log(y_out[E_idx+1] + 1),torch.zeros_like(y_out[E_idx+1])) #TMP, pt only hack 


    #relu =  m = nn.ReLU() #inplace=True  #This is actually not needed for E if min-max normalization is used for pt,E, AND!! relu is used as an activation function.
   # y_E_trimmed = relu(y[:,E_idx]) #trimming E
    #y_pt_trimmed = relu(pt) #trimming pt
   # full_y = torch.stack((y[:,PX_idx],y[:,PY_idx],y[:,PZ_idx],y_E_trimmed,y_pt_trimmed,eta,phi), dim=1)
   # full_y = torch.stack((y[:,PX_idx],y[:,PY_idx],y[:,PZ_idx],y[:,E_idx],y_pt_trimmed,eta,phi), dim=1)
    full_y = torch.stack(y_out, dim=1)

    return full_y


class LossFunction:
    def __init__(self, lossname, beta = 0.5,log_idx = [],device=torch.device('cuda:0')):
        loss = getattr(self, lossname)
        self.name = lossname
        self.loss_ftn = loss
        self.device = device
        self.beta = beta
        self.log_idx = log_idx
        
    def mse(self, y, x,reduction='mean'):#for some reason convension is : out,in
        PX_idx, PY_idx, PZ_idx, E_idx, PT_idx, ETA_idx, PHI_idx = range(7)
        #tricks for eta and phi
        y_phi = math.pi*torch.tanh(y[:,PHI_idx])
        y_eta = 2.5*torch.tanh(y[:,ETA_idx])
        full_y = torch.stack((y[:,PX_idx],y[:,PY_idx],y[:,PZ_idx],y[:,E_idx],y[:,PT_idx],y_eta,y_phi), dim=1)
        return F.mse_loss(x, y, reduction=reduction)
    
    def mse_coordinates(self, y,x,reduction='mean'): #for some reason convension is : out,in
        #From px,py,pz,E get pt, eta, phi (do not predict them)
        #x is px,py,pz,E,pt,eta,phi
        #y is px,py,pz
        full_y = xyze_to_ptetaphi_torch(y,log_idx = self.log_idx) 
        return self.mse(full_y,x,reduction=reduction)

    # Reconstruction + KL divergence losses
    def vae_loss_mse_coord(self, y,x, mu, logvar):
        MSE = self.mse_coordinates(y,x)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return (1-self.beta)*MSE + self.beta*KLD, MSE, KLD
        

    # Reconstruction + KL divergence losses
    def vae_loss_mse(self, x, y, mu, logvar):
        MSE = self.mse(y,x)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return (1-self.beta)*MSE + self.beta*KLD, MSE, KLD

    # Reconstruction + KL divergence losses
    def vae_flows_loss_mse_coord(self, x, y, mu, logvar,log_det_j, z_0, z_k):
        MSE = self.mse_coordinates(y,x,reduction='sum')
        return vae_flows_loss_mse(x, y, mu, logvar,log_det_j, z_0, z_k,MSE=MSE)

    def vae_flows_loss_mse(self, x, y, mu, logvar, log_det_j, z_0, z_k,MSE=None):
        batch_size = x.size(0)
        if MSE is None:
            MSE = self.mse(y,x,reduction='sum') #summming and will divide over batch

        # ln p(z_k)  (not averaged)
        log_p_zk = log_normal_standard(z_k, dim=1)
        # ln q(z_0)  (not averaged)
        log_q_z0 = log_normal_diag(z_0, mean=mu, log_var=logvar, dim=1)
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        summed_logs = torch.sum(log_q_z0 - log_p_zk)

        # sum over batches
        summed_ldj = torch.sum(log_det_j)
        # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
        kl = (summed_logs - summed_ldj)
        loss = MSE + kl

        loss = loss / int(batch_size)
        MSE = MSE / int(batch_size)
        kl = kl / int(batch_size)

        return loss, MSE, kl



