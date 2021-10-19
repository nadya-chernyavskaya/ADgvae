from pathlib import Path
import sys,os
import os.path as osp
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../vande/')))
sys.path.append(os.path.abspath(os.path.join('../../')))
sys.path.append(os.path.abspath(os.path.join('../DarkFlow/darkflow/')))
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
 epochs train_total_n valid_total_n gen_part_n batch_n learning_rate min_lr proc generator')
params = RunParameters(run_n=1, 
                       epochs=80, 
                       train_total_n=int(1e3 ),  #2e6 
                       valid_total_n=int(1e3), #1e5
                       gen_part_n=int(1e5), #1e5
                       batch_n=256, 
                       learning_rate=0.001,
                       min_lr=10e-6, 
                       proc='QCD_side',
                       generator=0)  #run generator or not

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)

# ********************************************************
#       Models params
# ********************************************************
Parameters = namedtuple('Settings', 'name  input_dim output_dim activation initializer big_dim hidden_dim beta loss_func')
settings = Parameters(name = 'PN',
                     input_dim=7,
                     output_dim=4,
                     activation='',#not yet set up
                     initializer='',#not yet set up 
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
print('>>> Preparing data')
start_time = time.time()
data_dir = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/output_models/pytroch/'
dataset = graph_data.GraphDataset(root=data_dir,n_jets=params.train_total_n)
# train (generator)
if params.generator:
    #to be filled
    print('Not yet prepared')
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
vande_plot.plot_features(pf_cands_t.numpy(), dataset.pf_kin_names_model ,'Normalized' , 'Jets Constituents', plotname='{}plot_pf_feats_{}'.format(fig_dir,params.proc), legend=[params.proc], ylogscale=True)
vande_plot.plot_features(jet_prop, dataset.jet_kin_names_model ,'Normalized' , 'Jets', plotname='{}plot_jet_feats_{}'.format(fig_dir,params.proc), legend=[params.proc], ylogscale=True)

# *******************************************************
#                       scaling input features 
# *******************************************************
minmax_idx = []
for f in 'E,pt':
    if f in dataset.pf_kin_names_model:
      minmax_idx.append(dataset.pf_kin_names_model.index(f))
log_idx = minmax_idx
scaler = prepr.standardize(in_memory_datas,minmax_idx=[3,4],log_idx=[3,4]) 
dataloaders = {
     'train':  DataLoader(in_memory_datas, batch_size=params.batch_n, shuffle=True)
}
# *******************************************************
#                       plotting input features after scaling
# *******************************************************
print('Plotting consistuents features after normalization')
pf_cands_norm = torch.cat([torch.tensor(pf_cands[i], dtype=torch.float) for i in range(len(in_memory_datas))])
#Plot consistuents and jet features prepared for the graph! (after normalization)
vande_plot.plot_features(pf_cands_norm.numpy(), dataset.pf_kin_names_model  ,'Jets Constituents Normalized' , params.proc, plotname='{}plot_pf_feats_norm_{}'.format(fig_dir,params.proc), legend=[params.proc], ylogscale=True)


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
#                       training options
# *******************************************************
optimizer = torch.optim.Adam(model.parameters(), lr = params.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=params.min_lr)
loss_ftn_obj = losses.LossFunction(settings.loss_func,beta=0.5,device=device)

# *******************************************************
#                       train and save
# *******************************************************
print('>>> Launching Training')

