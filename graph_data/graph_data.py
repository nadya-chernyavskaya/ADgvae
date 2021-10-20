import glob
import torch
import itertools
import numpy as np
import pandas as pd
import os.path as osp
import multiprocessing
from pathlib import Path
import h5py
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Dataset, Data, Batch
import scipy
from scipy.sparse import csr_matrix

def get_present_constit(x,n):
    return x[0:n,:] 

def concat_features(feats_1,feats_2):
    return np.hstack((feats_1[:,:],feats_2[:,:]))


class PairJetsData(Data):
    def __init__(self, edge_index_1=None, x_1=None, edge_index_2=None, x_2=None):
        super().__init__()
        self.edge_index_1 = edge_index_1
        self.x_1 = x_1
        self.edge_index_2 = edge_index_2
        self.x_2 = x_2

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
        

class GraphDataset(Dataset):  ####inherits from pytorch geometric Dataset (not just pytroch)
    def __init__(self, root, transform=None, pre_transform=None,
                 n_events=-1,n_jets=10e3, side_reg=1, proc_type='==0', features='xyzeptep',n_proc=1):
        """
        Initialize parameters of graph dataset
        Args:
            root (str): dir path
            n_events (int): how many events to process (-1=all in a file (there is a max))
            n_jets (int) : how many total jets to use
            side_reg (bool):true or false, side region for training, otherwise for testing on signal
            proc_type (str): string expression ==proc_type, or </>/~=/==   
            n_proc (int): number of processes to split into
            features (str): (px, py, pz) or relative (pt, eta, phi)
        """
        max_events = int(1.1e6)
        self.n_events = max_events if n_events==-1 else n_events
        self.n_jets = int(n_jets)
        self.side_reg = side_reg
        self.proc_type = proc_type
        self.n_proc = n_proc
        self.chunk_size = self.n_events // self.n_proc
        self.features = features
        self.dEtaJJ = 1.4
        self.jPt = 400
        self.jet_kin_names = ['mJJ', 'DeltaEtaJJ', 'j1Pt', 'j1Eta', 'j1Phi',\
                                        'j1M', 'j2Pt', 'j2Eta', 'j2Phi', 'j2M', 'j3Pt', 'j3Eta', 'j3Phi', 'j3M']
        self.pf_kin_names = 'px,py,pz,E,'.split(',')
        self.pf_kin_names_model = ''
        self.jet_kin_names_model = ''
        self.pf_cands, self.jet_prop = self.read_events()   

        
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        #self.input_files = sorted(glob.glob(self.raw_dir+'/*.root'))
        #return [f.split('/')[-1] for f in self.input_files]
        file = 'BB_batch0.h5'
        return file

    def __len__(self):
        return self.n_jets
        
  
    def xyze_to_ptep(self,constituents):
        ''' converts an array [N x 100, 4] of particles
        from px, py, pz, E to eta, phi, pt (mass omitted)
        '''
        PX = self.pf_kin_names.index('px')
        PY = self.pf_kin_names.index('py')
        PZ = self.pf_kin_names.index('pz')
        E = self.pf_kin_names.index('E')
        pt = np.sqrt(np.float_power(constituents[:,:,PX], 2) + np.float_power(constituents[:,:,PY], 2), dtype='float32') # numpy.float16 dtype -> float power to avoid overflow
        eta = np.arcsinh(np.divide(constituents[:,:,PZ], pt, out=np.zeros_like(pt), where=pt!=0.), dtype='float32')
        phi = np.arctan2(constituents[:,:,PY], constituents[:,:,PX], dtype='float32')
        return np.stack([pt, eta, phi], axis=2)



    def read_events(self):
        
        #This block bellow should be taken out as part of the reader
        #Data Samples
        DATA_PATH = '/eos/cms/store/group/phys_b2g/CASE/h5_files/full_run2/BB_UL_MC_small_v2/'
        TRAIN_NAME = 'BB_batch0.h5'
        filename_bg = DATA_PATH + TRAIN_NAME 
        in_file = h5py.File(filename_bg, 'r') 
        jet_kin = np.array(in_file["jet_kinematics"])
        truth = np.array(in_file["truth_label"])

        j1Pt_mask = (jet_kin[:,self.jet_kin_names.index('j1Pt')] > self.jPt)
        j2Pt_mask = (jet_kin[:,self.jet_kin_names.index('j2Pt')] > self.jPt)
        proc_mask = eval('truth[:,0]{}'.format(self.proc_type))
        full_mask = j1Pt_mask & j2Pt_mask & proc_mask #this also checks that there will be always two jets
        if self.side_reg : 
            full_mask = full_mask & (jet_kin[:,self.jet_kin_names.index('DeltaEtaJJ')] > self.dEtaJJ)
        else : 
            full_mask = full_mask & (jet_kin[:,self.jet_kin_names.index('DeltaEtaJJ')] < self.dEtaJJ)

        #Apply mask on jet kinematics, truth and pf cands
        jet_kin = jet_kin[full_mask][0:self.n_jets]
        truth = truth[full_mask][0:self.n_jets]
        jet_const = [np.array(in_file["jet1_PFCands"])[full_mask][0:self.n_jets],np.array(in_file["jet2_PFCands"])[full_mask][0:self.n_jets]]
        ###############                

        pf_out_list = []
        jet_prop_list = []

        for i_j in range(2): 
            pf_xyze = jet_const[i_j]
            pf_ptep = self.xyze_to_ptep(pf_xyze)
            n_particles = np.sum(pf_xyze[:,:,self.pf_kin_names.index('E')]!=0,axis=1) #E is 3rd 
            pf_xyze_out = list(map(get_present_constit,pf_xyze,n_particles))
            pf_ptep_out = list(map(get_present_constit,pf_ptep,n_particles))
            pf_tot_out = list(map(concat_features,pf_xyze_out,pf_ptep_out))
            pf_out_list.append(pf_tot_out)

            n_jet_feats = 6
            jet_prop = np.zeros((len(pf_tot_out),n_jet_feats))
            jet_prop[:,0] = n_particles
            for i_f,f_name in enumerate('M,Pt,Eta,Phi'.split(',')):
                jet_prop[:,i_f+1] = jet_kin[:,self.jet_kin_names.index('j{}{}'.format(i_j+1,f_name))]
            jet_prop[:,n_jet_feats-1] = truth[:,0]
            jet_prop_list.append(jet_prop)


            
        self.pf_kin_names_model = 'px,py,pz,E,pt,eta,phi'.split(',')
        self.jet_kin_names_model = 'N_constituents,M,Pt,Eta,Phi,truth'.split(',')

      # pf_interleave = pf_out_list[0] + pf_out_list[1]
      #  pf_interleave[::2] = pf_out_list[0]
      #  pf_interleave[1::2] =  pf_out_list[1]
        return list(itertools.chain(*zip(pf_out_list[0] , pf_out_list[1]))), list(itertools.chain(*zip(jet_prop_list[0] , jet_prop_list[1])))
        #or list(itertools.chain(*zip(pf_out_list[0] , pf_out_list[1])))
        #return list of pf particles, and list of global jet properties
        #return sum(pf_out_list, []),np.vstack((jet_prop_list[0],jet_prop_list[1]))      
                 

    def get(self,idx):
        '''Yields one data graph'''
        #pf_cands, jet_prop = self.read_events()  #if done like this, it will process the data each time - insane . Has to be rewritten/rethought with generator.
        
        i_evt = idx
        #for i_evt in range(len(pf_cands)):
        n_particles = self.pf_cands[i_evt].shape[0]
        pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles),range(n_particles)) if m!=n])
        edge_index = torch.tensor(pairs, dtype=torch.long)
        edge_index=edge_index.t().contiguous()
        # save particles as node attributes and target
        x = torch.tensor(self.pf_cands[i_evt], dtype=torch.float)
        u = torch.tensor(self.jet_prop[i_evt,:], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index,u=torch.unsqueeze(u, 0))
        return data
    
    def return_inmemory_data(self):
        datas = []
        n_jets = len(self.pf_cands)
        for i_evt in range(n_jets): 
            n_particles = self.pf_cands[i_evt].shape[0]
            adj = csr_matrix(np.ones((n_particles,n_particles)) - np.eye(n_particles)) 
            edge_index,_ = torch_geometric.utils.from_scipy_sparse_matrix(adj)          
            x = torch.tensor(self.pf_cands[i_evt], dtype=torch.float)
            u = torch.tensor(self.jet_prop[i_evt], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index,u=torch.unsqueeze(u, 0))
            datas.append(data)
        return datas
        
    def return_inmemory_data_no_loop(self):
        datas = []
        n_jets = len(self.pf_cands)
        n_particles = [self.pf_cands[i_evt].shape[0] for i_evt in range(n_jets)]
        adj = [csr_matrix(np.ones((n_part,n_part)) - np.eye(n_part)) for n_part in n_particles]
        edge_index = [from_scipy_sparse_matrix(a)[0] for a in adj] 
        # save particles as node attributes and target
        x= [torch.tensor(self.pf_cands[i_evt], dtype=torch.float) for i_evt in range(n_jets)] 
        u = [torch.tensor(self.jet_prop[i_evt], dtype=torch.float) for i_evt in range(n_jets)] 
        datas = [Data(x=x_jet, edge_index=edge_index_jet,u=torch.unsqueeze(u_jet, 0)) for x_jet,edge_index_jet,u_jet in zip(x,edge_index,u)]
        return datas
        

    def return_jets_pair_data(self):
        in_memory_datas = return_inmemory_data_no_loop(self)
        datas = [PairJetsData(in_memory_datas[i_evt].edge_index, in_memory_datas[i_evt].x, in_memory_datas[i_evt+1].edge_index, in_memory_datas[i_evt+1].x) for i_evt in range(0,self.n_jets,2)]
        return datas
        
