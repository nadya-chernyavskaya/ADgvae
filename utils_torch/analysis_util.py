import sys,os
import torch
import torch.nn.functional as F
import scipy.optimize
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
import math
import numpy as np
from collections import namedtuple
import time,pathlib
import h5py, json, pickle,glob, tqdm, math, random
import pandas as pd
from contextlib import redirect_stdout
import itertools
from itertools import chain



def get_mjj_binned_sample(df, mjj_peak, window_pct=20):
    left_edge, right_edge = mjj_peak * (1. - window_pct / 100.), mjj_peak * (1. + window_pct / 100.)

    center_bin_df = df[(df['dijet_mass'] >= left_edge) & (df['dijet_mass'] <= right_edge)]
    return center_bin_df


def get_df(proc_jets):
    d = {'loss_tot_1': proc_jets[:,0],
        'loss_reco_1': proc_jets[:,1],
        'loss_kl_1': proc_jets[:,2],
        'loss_tot_2': proc_jets[:,3],
        'loss_reco_2': proc_jets[:,4],
        'loss_kl_2': proc_jets[:,5],
        'dijet_mass': proc_jets[:,6],
        'mass1': proc_jets[:,7],
        'mass2': proc_jets[:,8],
        'truth_bit': proc_jets[:,9]}
    df = pd.DataFrame(d)
    return df



def eppt_to_xyz(m, pt, eta, phi):
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    e = torch.sqrt(torch.square(pt)+torch.square(pz)+torch.square(m))
    return e,px,py,pz



def invariant_mass_from_epxpypz(jet1_e, jet1_px, jet1_py, jet1_pz, jet2_e, jet2_px, jet2_py, jet2_pz):
    """
        Calculates the invariant mass between 2 jets. Based on the formula:
        m_12 = sqrt((E_1 + E_2)^2 - (p_x1 + p_x2)^2 - (p_y1 + p_y2)^2 - (p_z1 + p_z2)^2)

        Args:
            jet1_(e, px, py, pz) (torch.float): 4 momentum of first jet of dijet
            jet2_(e, px, py, pz) (torch.float): 4 momentum of second jet of dijet

        Returns:
            torch.float dijet invariant mass.
    """
    return torch.sqrt(torch.square(jet1_e + jet2_e) - torch.square(jet1_px + jet2_px)
                      - torch.square(jet1_py + jet2_py) - torch.square(jet1_pz + jet2_pz))
    
def invariant_mass_from_ptetaphim(jet1_m, jet1_pt, jet1_eta, jet1_phi, jet2_m, jet2_pt, jet2_eta, jet2_phi):
    """
        Calculates the invariant mass between 2 jets. Based on the formula:
        m_12 = sqrt((E_1 + E_2)^2 - (p_x1 + p_x2)^2 - (p_y1 + p_y2)^2 - (p_z1 + p_z2)^2).
    """
    jet1_e,jet1_px,jet1_py,jet1_pz = eppt_to_xyz(jet1_m, jet1_pt, jet1_eta, jet1_phi)
    jet2_e,jet2_px,jet2_py,jet2_pz = eppt_to_xyz(jet2_m, jet2_pt, jet2_eta, jet2_phi)
    return torch.sqrt(torch.square(jet1_e + jet2_e) - torch.square(jet1_px + jet2_px)
                      - torch.square(jet1_py + jet2_py) - torch.square(jet1_pz + jet2_pz))
    
def invariant_mass_from_ptetaphim_2(jet1_m, jet1_pt, jet1_eta, jet1_phi, jet2_m, jet2_pt, jet2_eta, jet2_phi): #probably a faster implementation
    jet1_et = torch.sqrt(torch.square(jet1_pt)+torch.square(jet1_m))
    jet2_et = torch.sqrt(torch.square(jet2_pt)+torch.square(jet2_m))
    return torch.sqrt(torch.square(jet1_m)+torch.square(jet2_m)+2*(jet1_et*jet2_et*torch.cosh(jet1_eta-jet2_eta)-jet1_pt*jet2_pt)) #assuming pseudorapidity and rapidity are the same given the practically 0 mass
    


def process(data_loader, model, loss_ftn_obj,jet_kin_names,device,save_every=5e6):
    """
    Use the specified model to determine the reconstruction loss of each sample.
    Also calculate the invariant mass of the jets.
    """
    model.eval()

    # Store the return values
    jets_proc_data,input_fts,reco_fts,truth_bit_fts = [],[],[],[]
    z_0_fts,z_last_fts,mu_fts,log_var_fts   = [],[],[],[]

    jets_proc_data_cpu,input_fts_cpu,reco_fts_cpu,truth_bit_fts_cpu = [],[],[],[]
    z_0_fts_cpu,z_last_fts_cpu,mu_fts_cpu,log_var_fts_cpu   = [],[],[],[]

    i_file=0
    event = 0
    # for each event in the dataset calculate the loss and inv mass for the leading 2 jets
    with torch.no_grad():
        for k, data_batch in tqdm.tqdm(enumerate(data_loader),total=len(data_loader)):
            multi_gpu = False #false for now
            if not multi_gpu:
                data_batch = data_batch.to(device)
            jets_x = data_batch.x
            batch = data_batch.batch
            jets_u = data_batch.u
            #dataset is such that leading and subleading jets are following each other (only events with 2 jets are considered for now)
            jets0_u = jets_u[::2] 
            jets1_u = jets_u[1::2]
            # run inference on all jets
            out = model(data_batch)#tuple
            if len(out)==6:
                jets_rec, mu, log_var, _, z_0, z_last = out
            elif len(out)==3:
                jets_rec, mu, log_var = out
            
            # calculate invariant mass (data.u format: p[event_idx, n_particles, jet.mass, jet.px, jet.py, jet.pz, jet.e]])
            idx_m,idx_pt,idx_eta,idx_phi = jet_kin_names.index('M'),jet_kin_names.index('Pt'),jet_kin_names.index('Eta'),jet_kin_names.index('Phi')
            idx_truth = jet_kin_names.index('truth')
            dijet_mass = invariant_mass_from_ptetaphim(jets0_u[:,idx_m], jets0_u[:,idx_pt], jets0_u[:,idx_eta], jets0_u[:,idx_phi], #e, px,py,pz
                                        jets1_u[:,idx_m], jets1_u[:,idx_pt], jets1_u[:,idx_eta], jets1_u[:,idx_phi])
            njets = len(torch.unique(batch))
            losses_tot = torch.zeros((njets), dtype=torch.float32,device=device)
            losses_reco = torch.zeros((njets), dtype=torch.float32,device=device)
            losses_kl = torch.zeros((njets), dtype=torch.float32,device=device)
            # calculate loss per each batch (jet)
            truth_bit_per_const = []
            for ib in torch.unique(batch):
                if  'vae_loss' in loss_ftn_obj.name:
                    losses_tot[ib],losses_reco[ib],losses_kl[ib] = loss_ftn_obj.loss_ftn(jets_rec[batch==ib], jets_x[batch==ib], mu[batch==ib], log_var[batch==ib])
                elif 'mse' in loss_ftn_obj.name:
                    losses_tot[ib] = loss_ftn_obj.loss_ftn(jets_rec[batch==ib], jets_x[batch==ib])
                #saving truth bit for features
                truth_bit_per_const.append(torch.full((jets_x[batch==ib].shape[0],),jets_u[int(ib.item()),idx_truth].item()))

            loss_tot_0,loss_reco_0,loss_kl_0 = losses_tot[::2],losses_reco[::2],losses_kl[::2]
            loss_tot_1,loss_reco_1,loss_kl_1 = losses_tot[1::2],losses_reco[1::2],losses_kl[1::2]
            #self.jet_kin_names_model = 'N_constituents,M,Pt,Eta,Phi,truth'.split(',')
            jets_info = torch.stack([loss_tot_0,loss_reco_0,loss_kl_0,
                                     loss_tot_1,loss_reco_1,loss_kl_1,
                                     dijet_mass,              # mass of dijet
                                     jets0_u[:,idx_m],            # mass of jet 1
                                     jets1_u[:,idx_m],            # mass of jet 2
                                     jets1_u[:,idx_truth]],        # the type of process
                                    dim=1)
            jets_proc_data.append(jets_info)
            input_fts.append(jets_x)
            reco_fts.append(jets_rec)
            mu_fts.append(mu)
            log_var_fts.append(log_var)
            if len(out)==6:
                z_0_fts.append(z_0)
                z_last_fts.append(z_last)
            truth_bit_per_const = torch.cat(truth_bit_per_const)
            truth_bit_fts.append(truth_bit_per_const)
            event += njets/2
            if (event>=save_every) or (k==len(data_loader)-1):
                jets_proc_data_cpu.append(torch.cat(jets_proc_data).cpu())
                input_fts_cpu.append(torch.cat(input_fts).cpu())
                reco_fts_cpu.append(torch.cat(reco_fts).cpu())
                z_0_fts_cpu.append(torch.cat(z_0_fts).cpu())
                z_last_fts_cpu.append(torch.cat(z_last_fts).cpu())
                mu_fts_cpu.append(torch.cat(mu_fts).cpu())
                log_var_fts_cpu.append(torch.cat(log_var_fts).cpu())
                truth_bit_fts_cpu.append(torch.cat(truth_bit_fts).cpu())
                del jets_proc_data,input_fts,reco_fts,truth_bit_fts
                del z_0_fts,z_last_fts,mu_fts,log_var_fts
                jets_proc_data,input_fts,reco_fts,truth_bit_fts = [],[],[],[]
                z_0_fts,z_last_fts,mu_fts,log_var_fts   = [],[],[],[]

        return np.vstack(jets_proc_data_cpu), np.vstack(input_fts_cpu), np.vstack(reco_fts_cpu),np.vstack(z_0_fts_cpu),np.vstack(z_last_fts_cpu),np.vstack(mu_fts_cpu),np.vstack(log_var_fts_cpu),np.vstack(truth_bit_fts_cpu)
         #   if (event>=save_every) or (k==len(data_loader)-1):
         #       df = get_df(torch.cat(jets_proc_data).cpu())
         #       df.to_pickle(osp.join(save_path,'predicted_df_{}_{}.pkl'.format(outname,ifile)))
         #       with h5py.File(osp.join(save_path, 'predicted_output_{}_{}.h5'.format(outname,ifile)), 'w') as outFile:
         #           outFile.create_dataset('reco_feats', data=torch.cat(reco_fts).cpu(), compression='gzip')
         #           outFile.create_dataset('input_fts', data=torch.cat(input_fts).cpu(), compression='gzip')
         #           outFile.create_dataset('z_0_fts', data=torch.cat(z_0_fts).cpu(), compression='gzip')
         #           outFile.create_dataset('z_last_fts', data=torch.cat(z_last_fts).cpu(), compression='gzip')
         #           outFile.create_dataset('mu_fts', data=torch.cat(mu_fts).cpu(), compression='gzip')
         #           outFile.create_dataset('log_var_fts', data=torch.cat(log_var_fts).cpu(), compression='gzip')
         #           outFile.create_dataset('truth_bit', data=torch.cat(truth_bit).cpu(), compression='gzip')
         #       jets_proc_data,input_fts,reco_fts,truth_bit_fts = [],[],[],[]
         #       z_0_fts,z_last_fts,mu_fts,log_var_fts   = [],[],[],[]
         #       i_file+=1

    # return pytorch tensors
    #return torch.cat(jets_proc_data), torch.cat(input_fts), torch.cat(reco_fts),torch.cat(z_0_fts),torch.cat(z_last_fts),torch.cat(mu_fts),torch.cat(log_var_fts),torch.cat(truth_bit_fts)
