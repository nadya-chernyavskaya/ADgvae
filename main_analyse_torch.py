from pathlib import Path
import sys,os
import os.path as osp
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../vande/')))
sys.path.append(os.path.abspath(os.path.join('../sarewt_orig/')))
sys.path.append(os.path.abspath(os.path.join('../../')))
sys.path.append(os.path.abspath(os.path.join('../DarkFlow/darkflow/')))
import vande.util.util_plotting as vande_plot
import vande.analysis.analysis_roc as ar
import pofah.util.experiment as expe

import models_torch.models as models
import models_torch.losses as torch_losses
import utils_torch.scaler
import utils_torch.preprocessing as prepr
import utils_torch.plot_util as plot
import utils_torch.train_util as train
import utils_torch.analysis_util as analysis
import graph_data.graph_data_h5py as graph_data
import utils_torch.model_summary as summary

import numpy as np
from collections import namedtuple
import time,pathlib
import h5py, json, pickle,glob, tqdm, math, random
import pandas as pd
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import EdgeConv, global_mean_pool, DataParallel
import setGPU

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on the device ',device)
multi_gpu = False #torch.cuda.device_count()>1
num_workers = 0



# ********************************************************
#       runtime params
# ********************************************************
RunParameters = namedtuple('Parameters', 'run_n  test_total_n ') 
in_params = RunParameters(run_n=22, test_total_n=int(4e5)) 
experiment = expe.Experiment(in_params.run_n).setup(model_dir=True, fig_dir=True)

with open(os.path.join(experiment.model_dir,'parameters.json'), 'r') as f_json:
    saved_params = json.loads(json.load(f_json))
saved_params['test_total_n'] = in_params.test_total_n
del saved_params['proc']#removing process infromation on training
with open(os.path.join(experiment.model_dir,'scaler.pkl'), 'rb') as outp:
    scaler = pickle.load(outp)
RunParameters = namedtuple('RunParameters', saved_params)
params = RunParameters(**saved_params)

loss_ftn_obj = torch_losses.LossFunction(params.loss_func)
model_path = osp.join(experiment.model_dir, params.model_name+'.best.pth')
model_fname = params.model_name
model =  getattr(models, params.model_name)(input_dim=params.input_dim,output_dim=params.output_dim, big_dim=params.big_dim, hidden_dim=params.hidden_dim)
model.load_state_dict(torch.load(model_path))
if multi_gpu:
    model = DataParallel(model)
model.to(device)


input_path = '/eos/cms/store/group/phys_b2g/CASE/h5_files/full_run2/BB_UL_MC_small_v2/'
root_path_sig_region_qcd = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_based/case_input/signal_region_qcd/'
root_path_sig_region_non_qcd_bg = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_based/case_input/signal_region_non_qcd_bg/'
root_path_sig_region_sig = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_based/case_input/signal_region_signals/'    
input_files_txt = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_based/case_input/signal_region_files.txt'
input_files_txt_qcd = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/autoencoder_for_anomaly/graph_based/case_input/signal_region_files_qcd.txt'
do_preprocessing=False
if do_preprocessing:
    print('>>> Preprocessing data')
    #Parameters for the graph dataset
    side_reg = 0
    proc_type='==0' #no process type #<0, ==0
    if proc_type=='<0':
        root_path_sig_region = root_path_sig_region_non_qcd_bg
    if proc_type=='==0':
        root_path_sig_region = root_path_sig_region_qcd
        input_files_txt = input_files_txt_qcd
    if proc_type=='>0':
        root_path_sig_region = root_path_sig_region_sig
    with open(input_files_txt,'r') as txt_file:
        lines = txt_file.readlines()
        input_files = [line.rstrip() for line in lines]
    #taking already processed files
    dataset = graph_data.GraphDataset(root=root_path_sig_region,input_path = input_path,input_files = input_files,proc_type=proc_type, n_events = params.test_total_n, side_reg=0,  scaler=scaler)
    exit()


procs_all = {}
procs_all['QCD'] = 0
procs_all['Graviton'] = 1
procs_all['W'] = 2
procs_all['Wkk'] = 3
procs_all['b*'] = 4
procs_all['Single Top'] = -1
procs_all['ttbar'] = -2
procs_all['V+jets'] = -3
procs_dict = {key: value for key, value in procs_all.items() if value >= 0}
consider_non_qcd_bg=False
procs_signals = {key: value for key, value in procs_all.items() if value > 0}
if consider_non_qcd_bg:
    procs_signals = {key: value for key, value in procs_all.items() if value != 0}
    procs_dict = procs_all



save_path = os.path.join(experiment.model_dir, 'predicted_data/')
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
overwrite = False
#if not osp.isfile(osp.join(save_path,'predicted_df_signal.pkl')) or overwrite:
if overwrite:
    print('>>> Preparing data')
    #taking already processed files
    signal_dataset = graph_data.GraphDataset(root=root_path_sig_region_sig,input_path = input_path, n_events = int(params.test_total_n), scaler=scaler)
    qcd_dataset = graph_data.GraphDataset(root=root_path_sig_region_qcd,input_path = input_path, n_events = int(params.test_total_n), scaler=scaler)
    non_qcd_bg_dataset = graph_data.GraphDataset(root=root_path_sig_region_non_qcd_bg,input_path = input_path, n_events = int(params.test_total_n), scaler=scaler)

    if multi_gpu:
        #shuffle is not going to work inside the DataLoaders, because of the way the Dataset is set up, shuffling option is passed to the dataset
        signal_loader = DataListLoader(signal_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)
        qcd_loader = DataListLoader(qcd_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)
        non_qcd_bg_loader = DataListLoader(non_qcd_bg_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)
    else:
        signal_loader = DataLoader(signal_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)
        qcd_loader = DataLoader(qcd_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)
        non_qcd_bg_loader = DataLoader(non_qcd_bg_dataset, batch_size=params.batch_n, num_workers=num_workers, pin_memory=True, shuffle=False)

    jet_kin_names = signal_dataset.jet_kin_names_model
    #for loader,name in zip([signal_loader,qcd_loader,non_qcd_bg_loader],['signal','qcd','non_qcd_bg']):
    for loader,name in zip([signal_loader,non_qcd_bg_loader],['signal','non_qcd_bg']):
        proc_jets, input_fts, reco_fts, z_0_fts,z_last_fts,mu_fts,log_var_fts,truth_bit = analysis.process(loader, model, loss_ftn_obj,jet_kin_names,device)
        df = analysis.get_df(proc_jets)
        df.to_pickle(osp.join(save_path,'predicted_df_{}.pkl'.format(name)))
        with h5py.File(osp.join(save_path, 'predicted_output_{}.h5'.format(name)), 'w') as outFile:
            outFile.create_dataset('reco_feats', data=reco_fts, compression='gzip')
            outFile.create_dataset('input_fts', data=input_fts, compression='gzip')
            outFile.create_dataset('z_0_fts', data=z_0_fts, compression='gzip')
            outFile.create_dataset('z_last_fts', data=z_last_fts, compression='gzip')
            outFile.create_dataset('mu_fts', data=mu_fts, compression='gzip')
            outFile.create_dataset('log_var_fts', data=log_var_fts, compression='gzip')
            outFile.create_dataset('truth_bit', data=truth_bit, compression='gzip')

    exit()
else:
    print("Using preprocessed dictionary")
    fig_dir = os.path.join(experiment.model_dir, 'predicted_figs/')
    df_signal = pd.read_pickle(osp.join(save_path,'predicted_df_signal.pkl'))
    df_qcd = pd.read_pickle(osp.join(save_path,'predicted_df_qcd.pkl'))
    pred_sig = h5py.File(osp.join(save_path,'predicted_output_signal.h5'),'r',driver='core',backing_store=False)
    pred_qcd = h5py.File(osp.join(save_path,'predicted_output_qcd.h5'),'r',driver='core',backing_store=False)
    if consider_non_qcd_bg:
        df_non_qcd_bg = pd.read_pickle(osp.join(save_path,'predicted_df_non_qcd_bg.pkl'))
        df_signal = pd.concat([df_signal,df_non_qcd_bg])
        pred_sig = h5py.File(osp.join(save_path,'predicted_output_sig_non_qcd_bg.h5'),'r',driver='core',backing_store=False)
        fig_dir = os.path.join(experiment.model_dir, 'predicted_figs_all_samples/')
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)


    losses = ['loss_reco'] 
    if 'vae'.lower() in model_fname.lower():
        losses+=['loss_tot','loss_kl']

    feature_format = 'all_mse' #should be configurable above
    mu_datas,log_var_data = [],[]
    for proc, proc_bit in procs_dict.items(): 
        input_file = pred_sig if proc!='QCD' else pred_qcd
        mask = np.where(np.array(input_file['truth_bit']).astype(int)==proc_bit)[0]
        if proc=='QCD' : mask = np.where(np.array(input_file['truth_bit']).T.astype(int)==proc_bit)[0] #to be removed later
        plot_stat = int(2e4)
        plot.plot_reco(np.array(input_file['input_fts'])[mask][0:plot_stat], np.array(input_file['reco_feats'])[mask][0:plot_stat],scaler, True, model_fname, osp.join(osp.join(fig_dir,'signals'),proc.replace(' ','_')), feature_format,title=proc)
        plot.plot_latent(np.array(input_file['z_0_fts'])[mask][0:plot_stat],np.array(input_file['z_last_fts'])[mask][0:plot_stat],osp.join(osp.join(fig_dir,'signals'),proc.replace(' ','_')),title=proc)
        #plot mu and sigma
        mu = np.array(input_file['mu_fts'])[mask][0:plot_stat]
        log_var = np.array(input_file['log_var_fts'])[mask][0:plot_stat]
        mu_datas.append(mu)
        log_var_data.append(log_var)
        #jets
        input_df = df_signal if proc!='QCD' else df_qcd
        for loss in losses:
            input_df['{}_sum'.format(loss)] = input_df['{}_1'.format(loss)]+input_df['{}_2'.format(loss)]
            input_df['{}_min'.format(loss)] = np.minimum(input_df['{}_1'.format(loss)],input_df['{}_2'.format(loss)])
            input_df['{}_max'.format(loss)] = np.maximum(input_df['{}_1'.format(loss)],input_df['{}_2'.format(loss)])
    vande_plot.plot_features(np.array(mu_datas),'Gauss mu'  ,'Normalized' , '', plotname='{}plot_mu'.format(fig_dir), legend=list(procs_dict.keys()), ylogscale=True)
    vande_plot.plot_features(np.array(log_var_data), 'Gauss log sigma'  ,'Normalized' , '', plotname='{}plot_log_var'.format(fig_dir), legend=list(procs_dict.keys()), ylogscale=True)



    print('Plotting losses')
    for loss in losses:
        for jet in '1,2'.split(','):
            datas = []
            for proc, proc_bit in procs_dict.items(): 
                input_df = df_signal if proc!='QCD' else df_qcd
                mask = input_df['truth_bit'].astype(int)==proc_bit
                datas.append(input_df['{}_{}'.format(loss,jet)][mask])
            vande_plot.plot_hist_many(datas, '{}, jet {}'.format(loss,jet).replace('_',' ') ,'Normalized Dist.' , '', plotname='{}plot_{}_jet{}_log'.format(fig_dir,loss,jet), legend=list(procs_dict.keys()), ylogscale=True)
            vande_plot.plot_hist_many(datas, '{}, jet {}'.format(loss,jet).replace('_',' ') ,'Normalized Dist.' , '', plotname='{}plot_{}_jet{}'.format(fig_dir,loss,jet), legend=list(procs_dict.keys()), ylogscale=False)

    print('Plotting ROCs')
    for loss in losses:
        for comb in 'sum,min,max'.split(','):
            mask_qcd = df_qcd['truth_bit'].astype(int)==procs_dict['QCD']
            neg_class_losses = [df_qcd['{}_{}'.format(loss,comb)][mask_qcd]]*len(list(procs_signals.keys()))
            pos_class_losses = []
            for proc, proc_bit in procs_signals.items(): 
                input_df = df_signal 
                mask = input_df['truth_bit'].astype(int)==proc_bit
                pos_class_losses.append(input_df['{}_{}'.format(loss,comb)][mask])
            ar.plot_roc( neg_class_losses, pos_class_losses, legend=list(procs_signals.keys()), title='{}'.format('{}_{}'.format(loss,comb)),
            plot_name='ROC_{}_{}'.format(loss,comb), fig_dir=fig_dir,log_x=False )

    print('Plotting binned in Mjj ROCs')
    for loss in losses:
        for comb in 'sum,min,max'.split(','):
            mass_center=2500
            neg_class_losses = []
            pos_class_losses = []
            for proc, proc_bit in procs_signals.items(): 
                if proc=='b*' :
                    mass_center==2600 
                else : 
                    mass_center = 2500
                mask = df_signal['truth_bit'].astype(int)==proc_bit
                pos = df_signal[mask]
                pos = analysis.get_mjj_binned_sample(pos,mass_center)
                pos = pos['{}_{}'.format(loss,comb)]
                pos_class_losses.append(pos)

                mask_qcd = df_qcd['truth_bit'].astype(int)==procs_dict['QCD']
                neg = df_qcd[mask_qcd]
                neg = analysis.get_mjj_binned_sample(neg,mass_center)
                neg = neg['{}_{}'.format(loss,comb)]
                neg_class_losses.append(neg)
            ar.plot_roc( neg_class_losses, pos_class_losses, legend=list(procs_signals.keys()), title='{}'.format('{}_{}, bin mJJ '.format(loss,comb)),
            plot_name='ROC_{}_{}_mjj_binned'.format(loss,comb), fig_dir=fig_dir,log_x=False )


    pred_sig.close()
    pred_qcd.close()




