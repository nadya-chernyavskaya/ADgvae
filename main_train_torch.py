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
 n_epochs train_total_n valid_total_n gen_part_n batch_n learning_rate min_lr patience proc generator')
params = RunParameters(run_n=1, 
                       n_epochs=3, 
                       train_total_n=int(1e3 ),  #2e6 
                       valid_total_n=int(1e3), #1e5
                       gen_part_n=int(1e5), #1e5
                       batch_n=256, 
                       learning_rate=0.001,
                       min_lr=10e-6,
                       patience=4,
                       proc='QCD_side',
                       generator=0)  #run generator or not

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)

# ********************************************************
#       Models params
# ********************************************************
Parameters = namedtuple('Settings', 'model_name  input_dim output_dim activation initializer big_dim hidden_dim beta loss_func')
settings = Parameters(model_name = 'PlanarEdgeNetVAE',
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
scaler = prepr.standardize(in_memory_datas,minmax_idx=minmax_idx,log_idx=log_idx) 
dataloaders = {
     'train':  DataLoader(in_memory_datas, batch_size=params.batch_n, shuffle=True)
}
# *******************************************************
#                       plotting input features after scaling
# *******************************************************
print('Plotting consistuents features after normalization')
pf_cands_norm = torch.cat([torch.tensor(in_memory_datas[i].x, dtype=torch.float) for i in range(len(in_memory_datas))])
#Plot consistuents and jet features prepared for the graph! (after normalization)
vande_plot.plot_features(pf_cands_norm.numpy(), dataset.pf_kin_names_model  ,'Normalized' , 'Jets Constituents Normalized', plotname='{}plot_pf_feats_norm_{}'.format(fig_dir,params.proc), legend=[params.proc], ylogscale=True)


# *******************************************************
#                       build model
# *******************************************************
model = getattr(models, settings.model_name)(input_dim=settings.input_dim,output_dim=settings.output_dim, big_dim=settings.big_dim, hidden_dim=settings.hidden_dim)


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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params.patience, threshold=params.min_lr)
loss_ftn_obj = losses.LossFunction(settings.loss_func,beta=0.5,device=device)

# *******************************************************
#                       train and save
# *******************************************************
print('>>> Launching Training')
# Training loop
stale_epochs = 0
loss = 999999

train_loader, train_samples = dataloaders['train'], len(dataloaders['train'].dataset)
valid_loader, valid_samples = dataloaders['train'], len(dataloaders['train'].dataset)
test_loader, valid_samples = dataloaders['train']

train_losses, valid_losses = {},{}
for what in 'tot,reco,kl'.split(','):
    train_losses[what] = []
    valid_losses[what] = []
start_epoch = 0
modpath = osp.join(experiment.model_dir, settings.model_name+'.best.pth')
if osp.isfile(modpath):
    model.load_state_dict(torch.load(modpath, map_location=device))
    model.to(device)
    best_valid_loss,best_valid_loss_reco,best_valid_kl = train.test(model, valid_loader, valid_samples, params.batch_n, loss_ftn_obj)
    print('Loaded model')
    print(f'Saved model valid loss tot, reco, kl: {best_valid_loss,best_valid_loss_reco,best_valid_kl}')
    if osp.isfile(osp.join(experiment.model_dir,'losses.pt')):
        train_losses, valid_losses, start_epoch = torch.load(osp.join(experiment.model_dir,'losses.pt'))
else:
    print('Creating new model')
    best_valid_loss = 9999999
    model.to(device)
if multi_gpu:
    model = DataParallel(model)
    model.to(device)

# Training loop
stale_epochs = 0
loss = best_valid_loss
epoch=0
for epoch in range(start_epoch, params.n_epochs):
    loss,loss_reco,loss_kl = train.train(model, optimizer, train_loader, train_samples, params.batch_n, loss_ftn_obj)
    valid_loss,valid_loss_reco,valid_loss_kl = train.test(model, valid_loader, valid_samples, params.batch_n, loss_ftn_obj)

    scheduler.step(valid_loss)
    train_losses['tot'].append(loss)
    valid_losses['tot'].append(valid_loss)
    train_losses['reco'].append(loss_reco)
    valid_losses['reco'].append(valid_loss_reco)
    train_losses['kl'].append(loss_kl)
    valid_losses['kl'].append(valid_loss_kl)
    print('Epoch: {:02d}, Training Loss Tot, Reco, KL :  {:.4f},{:.4f}, {:.4f}'.format(epoch, loss,loss_reco,loss_kl))
    print('Epoch: {:02d}, Validation Loss Tot, Reco, KL :  {:.4f},{:.4f}, {:.4f}'.format(epoch, valid_loss,valid_loss_reco,valid_loss_kl))

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print('New best model saved to:',modpath)
        if multi_gpu:
            torch.save(model.module.state_dict(), modpath)
        else:
            torch.save(model.state_dict(), modpath)
        torch.save((train_losses, valid_losses, epoch+1), osp.join(experiment.model_dir,'losses.pt'))
        stale_epochs = 0
    else:
        stale_epochs += 1
        print(f'Stale epoch: {stale_epochs}\nBest: {best_valid_loss}\nCurr: {valid_loss}')
    if stale_epochs >= params.patience:
        print('Early stopping after %i stale epochs'%params.patience)
        break

# model training done
if epoch!=0:
    train_epochs = list(range(epoch+1))
    early_stop_epoch = epoch - stale_epochs
    for what in 'tot,reco,kl'.split(','):
     plot.loss_curves(train_epochs, early_stop_epoch, train_losses[what], valid_losses[what], experiment.model_dir, fig_name=what)

# load best model
del model
torch.cuda.empty_cache()
model =  getattr(models, settings.model_name)(input_dim=settings.input_dim,output_dim=settings.output_dim, big_dim=settings.big_dim, hidden_dim=settings.hidden_dim)
model.load_state_dict(torch.load(modpath))
if multi_gpu:
    model = DataParallel(model)
model.to(device)

print('Plotting input/output reco')
inverse_standardization = True
plot_scale = 'all_mseconv'
plot.plot_reco_for_loader(model, train_loader, device, scaler, inverse_standardization, settings.model_name, osp.join(fig_dir, 'reconstruction_post_train', 'train'), plot_scale)
plot.plot_reco_for_loader(model, valid_loader, device, scaler, inverse_standardization, settings.model_name, osp.join(fig_dir, 'reconstruction_post_train', 'valid'), plot_scale)
plot.plot_reco_for_loader(model, test_loader, device, scaler, inverse_standardization, settings.model_name, osp.join(fig_dir, 'reconstruction_post_train', 'test'), plot_scale)
print('Completed')