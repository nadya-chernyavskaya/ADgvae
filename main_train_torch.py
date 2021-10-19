from pathlib import Path
import sys,os
import os.path as osp
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../vande/')))
sys.path.append(os.path.abspath(os.path.join('../../')))
import vande.util.util_plotting as vande_plot
import pofah.util.experiment as expe

import models_torch.models as models
import models_torch.losses as losses
import utils_torch.scaler
import utils_torch.preprocessing as prepr
import utils_torch.plot_util as plot
import utils_torch.train_util as train
import graph_data.graph_data as graph_data
import utils_torch.model_summary as summary

import setGPU
import numpy as np
from collections import namedtuple
import time,pathlib
import h5py, json, glob, tqdm, math, random
from contextlib import redirect_stdout
import itertools
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/utils/adfigstyle.mplstyle')

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.nn import EdgeConv, global_mean_pool, DataParallel

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multi_gpu = False #torch.cuda.device_count()>1

# ********************************************************
#       runtime params
# ********************************************************
RunParameters = namedtuple('Parameters', 'run_n  \
 epochs train_total_n gen_part_n valid_total_n batch_n learning_rate max_lr_decay lambda_reg generator')
params = RunParameters(run_n=1, 
                       epochs=80, 
                       train_total_n=int(1e3 ),  #2e6 
                       valid_total_n=int(1e3), #1e5
                       gen_part_n=int(1e5), #1e5
                       batch_n=256, 
                       learning_rate=0.001,
                       min_lr=10e-6, 
                       generator=0)  #run generator or not

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)

# ********************************************************
#       Models params
# ********************************************************
Parameters = namedtuple('Settings', 'name  input_shape output_shape activation initializer big_dim hidden_dim beta loss_func')
settings = Parameters(name = 'PN',
                     input_shape=7,
                     output_shape=4,
                     activation='',#needs to be filled
                     initializer='',#needs to be filled 
                     big_dim=32,
                     hidden_dim=2,
                     beta=0.5,
                     loss_func = 'vae_loss_mse_coord')


''' saving model parameters''' 
SetupParameters = namedtuple("SetupParameters", RunParameters._fields + Parameters._fields)
save_params = SetupParameters(*(params + settings))
saev_params_json = json.dumps((save_params._replace(activation='activation'))._asdict()) #replacing activation as you cannot save it
with open(os.path.join(experiment.model_dir,'parameters.json'), 'w', encoding='utf-8') as f_json:
    json.dump(saev_params_json, f_json, ensure_ascii=False, indent=4)

# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************
print('>>> Launching Training')
start_time = time.time()
data_dir = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/output_models/pytroch/'
dataset = graph_data.GraphDataset(root=data_dir,n_jets=params.train_total_n)
# train (generator)
if params.generator:
    #to be filled
else : 
    in_memory_datas = dataset.return_inmemory_data_no_loop() 
# *******************************************************
#                       plotting input features before scaling
# *******************************************************
print('Plotting consistuents features before normalization')
fig_dir = os.path.join(experiment.model_dir, 'figs/')
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

pf_cands,jet_prop = dataset.pf_cands,dataset.jet_prop
pf_cands_t = torch.cat([torch.tensor(pf_cands[i], dtype=torch.float) for i in range(len(pf_cands))])
#Plot consistuents and jet features prepared for the graph! (but before any normalization)
pf_feats = 'px,py,pz,E,pt,eta,phi'.split(',')
jet_feats = 'N_constituents,M,Pt,Eta,Phi'.split(',')
vande_plot.plot_features(pf_cands_t.numpy(), pf_feats ,'Normalized' , 'QCD', plotname='{}plot_pf_feats_{}'.format(fig_dir,'QCD_side'), legend=['QCD'], ylogscale=True)
vande_plot.plot_features(jet_prop[:,0:-1], jet_feats ,'Normalized' , 'QCD', plotname='{}plot_jet_feats_{}'.format(fig_dir,'QCD_side'), legend=['QCD'], ylogscale=True)

# *******************************************************
#                       scaling input features 
# *******************************************************
scaler = prepr.standardize(in_memory_datas,minmax_idx=[3,4],log_idx=[3,4]) 
dataloaders = {
     'train':  DataLoader(in_memory_datas, batch_size=128, shuffle=True)
}
# *******************************************************
#                       plotting input features after scaling
# *******************************************************
print('Plotting consistuents features after normalization')
pf_cands_norm = torch.cat([torch.tensor(in_memory_datas[i].x, dtype=torch.float) for i in range(len(in_memory_datas))])
#Plot consistuents and jet features prepared for the graph! (after normalization)
vande_plot.plot_features(pf_cands_norm.numpy(), pf_feats ,'Normalized' , 'QCD', plotname='{}plot_pf_feats_norm_{}'.format(save_dir,'QCD_side'), legend=['QCD'], ylogscale=True)


# *******************************************************
#                       training options
# *******************************************************

optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=params.min_lr)
loss_ftn_obj = losses.LossFunction(settings.loss_func,beta=0.5,device=device)

# *******************************************************
#                       build model
# *******************************************************
model = models.PlanarEdgeNetVAE(input_dim=settings.input_dim,output_dim=settings.output_dim, big_dim=settings.big_dim, hidden_dim=settings.hidden_dim)

model.to(device)

print(model)
summary.gnn_model_summary(model)
with open(os.path.join(experiment.model_dir,'model_summary.txt'), 'w') as f:
    with redirect_stdout(f):
        print(model)
        print(summary.gnn_model_summary(model))

# *******************************************************
#                       train and save
# *******************************************************

