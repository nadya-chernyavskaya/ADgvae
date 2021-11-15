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
import utils_torch.scaler as uscaler
import utils_torch.preprocessing as prepr
import utils_torch.plot_util as plot
import utils_torch.train_util as train
import graph_data.graph_data_h5py as graph_data
import utils_torch.model_summary as summary

import numpy as np
from collections import namedtuple
import time,pathlib
import h5py, json,pickle, glob, tqdm, math, random
from contextlib import redirect_stdout
import itertools
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/utils/adfigstyle.mplstyle')

import torch
import torch.nn as nn
from torch.utils.data import random_split, ConcatDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import EdgeConv, global_mean_pool, DataParallel
from torch.utils.tensorboard import SummaryWriter
import setGPU

torch.manual_seed(0)
device = torch.device('cuda:{}'.format(os.environ['CUDA_VISIBLE_DEVICES']) if torch.cuda.is_available() else 'cpu')
multi_gpu = False #torch.cuda.device_count()>1 #using multi-gpu is not tested with the current DataSet and might not work
print('Running on the device ',device,', Using multigpu ',multi_gpu)
num_workers = 0

# ********************************************************
#       runtime params
# ********************************************************
RunParameters = namedtuple('Parameters', 'run_n  \
 n_epochs train_total_n valid_total_n proc batch_n learning_rate min_lr patience min_delta adam_betas plotting generator')
params = RunParameters(run_n=25, 
                       n_epochs=50, 
                       train_total_n=int(1e6 ),  #1e6 
                       valid_total_n=int(2e5), #1e5
                       proc='QCD_side',
                       batch_n=200, 
                       learning_rate=0.0001,
                       min_lr=10e-8,
                       patience=6,
                       min_delta=0.01, #the larger the value, the less sensitive it is 
                       adam_betas=(0.7,0.9), #0.7, 0.9 #default (0.9, 0.999)
                       plotting=False,
                       generator=1) 

#Parameters for the graph dataset
if 'QCD_side' in params.proc:
    side_reg = 1
    proc_type='==0'
input_path = '/eos/cms/store/group/phys_b2g/CASE/h5_files/full_run2/BB_UL_MC_small_v2/'
root_path_train = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_based/case_input/train/'
root_path_valid = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_based/case_input/valid/'    
do_preprocessing=False
if do_preprocessing:
    input_files_train_txt = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_based/case_input/qcd_train_files.txt'
    input_files_valid_txt = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_based/case_input/qcd_valid_files.txt'
    print('>>> Preprocessing data')
    #Parameters for the graph dataset
    with open(input_files_train_txt,'r') as txt_file:
        lines = txt_file.readlines()
        input_files = [line.rstrip() for line in lines]
    #taking already processed files
    train_dataset = graph_data.GraphDataset(root=root_path_train,input_path = input_path,input_files = input_files,proc_type=proc_type, n_events = params.test_total_n, side_reg=side_reg,  scaler=scaler)
    with open(input_files_valid_txt,'r') as txt_file:
        lines = txt_file.readlines()
        input_files = [line.rstrip() for line in lines]
    valid_dataset = graph_data.GraphDataset(root=root_path_train,input_path = input_path,input_files = input_files,proc_type=proc_type, n_events = params.test_total_n, side_reg=side_reg,  scaler=scaler)
    exit()

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)

# ********************************************************
#       Models params
# ********************************************************
Parameters = namedtuple('Settings', 'model_name  input_dim output_dim loss_func standardizer big_dim hidden_dim beta num_flows activation initializer')
settings = Parameters(model_name = 'PlanarEdgeNetVAE',#'TriangularSylvesterEdgeNetVAE',
                     input_dim=7,
                     output_dim=7, #3/4 or 7 
                     loss_func = 'vae_loss_mse',  #  vae_flows_loss_mse_coord vae_loss_mse vae_loss_mse_coord',
                     standardizer=uscaler.BasicStandardizer(),  
                     big_dim=62,
                     hidden_dim=2,
                     beta=0.5,
                     num_flows=20,
                     activation=nn.ReLU(), #nn.LeakyReLU(0.1), #nn.ELU(),#nn.ReLU(),
                     initializer='') #not yet set up 



# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************
print('>>> Preparing data')
start_time = time.time()
#taking already processed files
train_dataset = graph_data.GraphDataset(root=root_path_train,input_path = input_path, n_events = params.train_total_n, scaler=settings.standardizer,shuffle=True)
valid_dataset = graph_data.GraphDataset(root=root_path_valid,input_path = input_path, n_events = params.valid_total_n, scaler=settings.standardizer)
if not (params.generator):
    #Loading in memory data, but only from the first file
    train_dataset.data_chunk_size = params.train_total_n
    train_dataset = train_dataset.in_memory_data(shuffle=True)
    valid_dataset.data_chunk_size = params.valid_total_n
    valid_dataset = valid_dataset.in_memory_data(shuffle=False)
train_samples = len(train_dataset)
valid_samples = len(valid_dataset)
print(f"Total number of train/valid events : {train_samples,valid_samples}")

if multi_gpu:
    #shuffle is not going to work inside the DataLoaders, because of the way the Dataset is set up, shuffling option is passed to the dataset
    train_loader = DataListLoader(train_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)
    valid_loader = DataListLoader(valid_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)
else:
    train_loader = DataLoader(train_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)

# ********************************************************
#       saving model parameters
# ********************************************************
SetupParameters = namedtuple("SetupParameters", RunParameters._fields + Parameters._fields)
save_params = SetupParameters(*(params + settings))
save_params_json = json.dumps((save_params._replace(activation='activation')._replace(standardizer=save_params.standardizer.name))._asdict()) #replacing activation as you cannot save it, but it is saved in the model file
with open(os.path.join(experiment.model_dir,'parameters.json'), 'w', encoding='utf-8') as f_json:
    json.dump(save_params_json, f_json, ensure_ascii=False, indent=4)
with open(os.path.join(experiment.model_dir,'scaler.pkl'), 'wb') as outp:
    pickle.dump(train_dataset.scaler, outp, pickle.HIGHEST_PROTOCOL)

# *******************************************************
#                       plotting input features before scaling
# *******************************************************
fig_dir = os.path.join(experiment.model_dir, 'figs/')
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
if params.plotting:
    len_plot = int(1e4)
    plot_dataset = graph_data.GraphDataset(root=root_path_train,input_path = input_path, n_events = len_plot,scaler=None)
    graph_data.data_chunk_size = len_plot
    print('>>> Plotting consistuents features before normalization')
    pf_cands,jet_prop =  plot_dataset.get_pfcands_jet_prop()
    vande_plot.plot_features(np.concatenate(pf_cands), plot_dataset.pf_kin_names_model  ,'Normalized' , 'Jets Constituents', plotname='{}plot_pf_feats_{}'.format(fig_dir,params.proc), legend=[params.proc], ylogscale=True)
    vande_plot.plot_features(jet_prop, plot_dataset.jet_kin_names_model ,'Normalized' , 'Jets', plotname='{}plot_jet_feats_{}'.format(fig_dir,params.proc), legend=[params.proc], ylogscale=True)
    print('>>> Plotting consistuents features after normalization')
    plot_dataset = graph_data.GraphDataset(root=root_path_train,input_path = input_path, n_events = len_plot,scaler=settings.standardizer)
    pf_cands_norm,_ =  plot_dataset.get_pfcands_jet_prop()
    #Plot consistuents and jet features prepared for the graph! (after normalization)
    vande_plot.plot_features(np.concatenate(pf_cands_norm), plot_dataset.pf_kin_names_model  ,'Normalized' , 'Jets Constituents Normalized', plotname='{}plot_pf_feats_norm_{}'.format(fig_dir,params.proc), legend=[params.proc], ylogscale=True)

# *******************************************************
#                       build model
# *******************************************************
model = getattr(models, settings.model_name)(input_dim=settings.input_dim,output_dim=settings.output_dim, big_dim=settings.big_dim, hidden_dim=settings.hidden_dim, activation=settings.activation,num_flows=settings.num_flows)


print(model)
summary.gnn_model_summary(model)
with open(os.path.join(experiment.model_dir,'model_summary.txt'), 'w') as f:
    with redirect_stdout(f):
        print(model)
        print(summary.gnn_model_summary(model))

# *******************************************************
#                       training options
# *******************************************************
optimizer = torch.optim.Adam(model.parameters(), lr = params.learning_rate, betas=params.adam_betas)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=params.min_lr,verbose=True)
loss_ftn_obj = losses.LossFunction(settings.loss_func,beta=0.5,device=device,log_idx=train_dataset.scaler.idx_log)

# *******************************************************
#                       train and save
# *******************************************************
tb = SummaryWriter(log_dir=osp.join(experiment.model_dir,'tensorboard_logs/'))
input_data = next(iter(train_loader))

print('>>> Launching Training')
# Training loop
stale_epochs = 0
loss = 999999

train_losses, valid_losses = {},{}
for what in 'tot,reco,kl'.split(','):
    train_losses[what] = []
    valid_losses[what] = []
start_epoch = 0
modpath = osp.join(experiment.model_dir, settings.model_name+'.best.pth')
if osp.isfile(modpath):
    model.load_state_dict(torch.load(modpath, map_location=device))
    model.to(device)
    best_valid_loss,best_valid_loss_reco,best_valid_kl = train.test(model, valid_loader, valid_samples, params.batch_n, loss_ftn_obj,device,multi_gpu)
    print('Loaded model')
    print(f'Saved model valid loss tot, reco, kl: {best_valid_loss,best_valid_loss_reco,best_valid_kl}')
    if osp.isfile(osp.join(experiment.model_dir,'losses.pt')):
        train_losses, valid_losses, start_epoch = torch.load(osp.join(experiment.model_dir,'losses.pt'))
else:
    print('Creating new model')
    best_valid_loss = 9999999
    model.to(device)
if multi_gpu:
    model = DataParallel(model) #this  should be changed to DistributedDataParallel 
    model.to(device)

# Training loop
stale_epochs = 0
loss = best_valid_loss
epoch=0
#for epoch in range(0, 0):
for epoch in range(start_epoch, params.n_epochs):
    modpath_epoch = osp.join(osp.join(experiment.model_dir,'saved_models/'), settings.model_name+'.epoch_{}.pth'.format(epoch))
    loss,loss_reco,loss_kl = train.train(model, optimizer, train_loader, train_samples, params.batch_n, loss_ftn_obj,device,multi_gpu)
    valid_loss,valid_loss_reco,valid_loss_kl = train.test(model, valid_loader, valid_samples, params.batch_n, loss_ftn_obj,device,multi_gpu)

    scheduler.step(valid_loss)
    train_losses['tot'].append(loss)
    valid_losses['tot'].append(valid_loss)
    train_losses['reco'].append(loss_reco)
    valid_losses['reco'].append(valid_loss_reco)
    train_losses['kl'].append(loss_kl)
    valid_losses['kl'].append(valid_loss_kl)
    print('Epoch: {:02d}, Training Loss Tot, Reco, KL :  {:.4f},{:.4f}, {:.4f}'.format(epoch, loss,loss_reco,loss_kl))
    print('Epoch: {:02d}, Validation Loss Tot, Reco, KL :  {:.4f},{:.4f}, {:.4f}'.format(epoch, valid_loss,valid_loss_reco,valid_loss_kl))

    tb.add_scalar("Training Loss Tot", loss, epoch)
    tb.add_scalar("Training Loss Reco", loss_reco, epoch)
    tb.add_scalar("Training Loss KL", loss_kl, epoch)
    tb.add_scalar("Valid Loss Tot", valid_loss, epoch)
    tb.add_scalar("Valid Loss Reco", valid_loss_reco, epoch)
    tb.add_scalar("Valid Loss KL", valid_loss_kl, epoch)
    for layer_name, weight in model.named_parameters():
        tb.add_histogram(layer_name,weight, epoch)
        tb.add_histogram(f'{layer_name}.grad',weight.grad, epoch)

###### early stopping implemnetation for a decresing metric (loss) ####
    if params.min_delta!=0:
        if valid_loss - params.min_delta < best_valid_loss:
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
    print('Model saved to:',modpath_epoch) #saving model after each epoch
    if epoch>10:
        if multi_gpu:
            torch.save(model.module.state_dict(), modpath_epoch)
        else:
            torch.save(model.state_dict(), modpath_epoch)        

tb.close()



end_time = time.time()
print(f">>> Runtime of the training is {end_time - start_time}")

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

print('>>> Plotting input/output reco and latent space')
inverse_standardization = True
plot_scale = 'all_mseconv'
plot.plot_reco_latent_for_loader(model, train_loader, device, train_dataset.scaler, inverse_standardization, settings.model_name, osp.join(fig_dir, 'reconstruction_post_train', 'train'), plot_scale)
plot.plot_reco_latent_for_loader(model, valid_loader, device,train_dataset.scaler, inverse_standardization, settings.model_name, osp.join(fig_dir, 'reconstruction_post_train', 'valid'), plot_scale)
print('>>> Completed')

