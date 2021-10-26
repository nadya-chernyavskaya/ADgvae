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
from utils_torch.scaler import BasicStandardizer

def get_present_constit(x,n):
    return x[0:n,:] 

def concat_features(feats_1,feats_2):
    return np.hstack((feats_1[:,:],feats_2[:,:]))

def process_func(args):
    self, raw_path, k = args
    return self.process_one_chunk(raw_path, k)

# functions needed from original pytorch dataset class for overwriting _process ###
def to_list(x):
    if not isinstance(x, (tuple, list)) or isinstance(x, str):
        x = [x]
    return x

# augmented to be less robust but faster than original (remove check for all files)
def files_exist(files):
    return len(files) != 0

def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

def collate(items): # collate function for data loaders (transforms list of lists to list)
    l = sum(items, [])
    return Batch.from_data_list(l)


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
    def __init__(self, root,input_path = None,transform=None, pre_transform=None, train_not_test=1,
        n_events=-1, side_reg=1, proc_type='==0', features='xyzeptep',n_proc=1,scaler=BasicStandardizer()):
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
        self.strides = [0]
        self.scaler = scaler
        self.input_path = input_path if input_path is not None else '/eos/cms/store/group/phys_b2g/CASE/h5_files/full_run2/BB_UL_MC_small_v2/'
        self.train_not_test = train_not_test 
        max_events = int(5e6)
        self.n_events = max_events if n_events==-1 else int(n_events)
        self.side_reg = side_reg
        self.proc_type = proc_type
        self.n_proc = n_proc
        self.chunk_size = self.n_events // self.n_proc #not currently used
        self.features = features
        self.dEtaJJ = 1.4
        self.jPt = 200
        self.jet_kin_names = ['mJJ', 'DeltaEtaJJ', 'j1Pt', 'j1Eta', 'j1Phi',\
                                        'j1M', 'j2Pt', 'j2Eta', 'j2Phi', 'j2M', 'j3Pt', 'j3Eta', 'j3Phi', 'j3M']
        self.pf_kin_names = 'px,py,pz,E,'.split(',')
        self.pf_kin_names_model = 'px,py,pz,E,pt,eta,phi'.split(',')
        self.jet_kin_names_model = 'N_constituents,M,Pt,Eta,Phi,truth'.split(',')
        self.set_up_scaler()
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.current_file_idx=0
        self.current_in_file = h5py.File(self.processed_paths[self.current_file_idx],'r')

    @property
    def raw_dir(self) -> str: #overwrite
        return self.input_path

    @property 
    def processed_dir(self) -> str: #keep it as is in pytorch 
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        input_files_full = glob.glob(osp.join(self.raw_dir, '*.h5'))
        input_files = list(map(osp.basename, input_files_full))
        len_inp_files = len(input_files)
        if self.train_not_test :
            return [input_files[i_f] for i_f in range(0,2)] #first 2 
        else :
            return [input_files[i_f] for i_f in reversed(len_inp_files-2,len_inp_files)] #last 2  #this is just temp 
        return files

    @property
    def processed_file_names(self):
        #this has to be rewritten to have both testings files and training files
        """
        Returns a list of all the files in the processed files directory
        """
        #the best way is to just have train/valid files in different root_base directories to set up different loaders for train and validation
        proc_list = glob.glob(osp.join(self.processed_dir, 'BB*.h5'))
        return_list = list(map(osp.basename, proc_list))
        return return_list


    def len(self):
        #return self.strides[-1] #this will process all files
        if self.n_events <= self.strides[-1]:
            return self.n_events
        else:
            return self.strides[-1]

    def set_up_scaler(self):
        self.idx_gev = [self.pf_kin_names_model.index(f) for f in self.pf_kin_names_model if f in 'pt,E,px,py,pz'.split(',')]
        self.idx_coord = [self.pf_kin_names_model.index(f) for f in self.pf_kin_names_model if f in 'eta,phi'.split(',')]

    def download(self):
        # Download to `self.raw_dir`.
        pass  

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


    def process_one_chunk(self, raw_path, k, write_files=True):
        """
        Handles conversion of dataset file at raw_path into graph dataset.
        """
        in_file = h5py.File(raw_path, 'r') 
        #out_file = raw_path.split('/')[-1].replace('.h5','_{}.pt')
        out_file = raw_path.split('/')[-1].replace('.h5','_{}.h5')

        jet_kin = np.array(in_file["jet_kinematics"])[k * self.chunk_size:(k + 1) * self.chunk_size]
        truth = np.array(in_file["truth_label"])[k * self.chunk_size:(k + 1) * self.chunk_size]
        jet_const = [np.array(in_file["jet1_PFCands"])[k * self.chunk_size:(k + 1) * self.chunk_size],np.array(in_file["jet2_PFCands"])[k * self.chunk_size:(k + 1) * self.chunk_size]]

        j1Pt_mask = (jet_kin[:,self.jet_kin_names.index('j1Pt')] > self.jPt)
        j2Pt_mask = (jet_kin[:,self.jet_kin_names.index('j2Pt')] > self.jPt)
        full_mask = j1Pt_mask & j2Pt_mask #this also checks that there will be always two jets
        if self.proc_type is not None :
            proc_mask = eval('truth[:,0]{}'.format(self.proc_type))
            full_mask = full_mask & proc_mask 

        if self.side_reg : 
            full_mask = full_mask & (jet_kin[:,self.jet_kin_names.index('DeltaEtaJJ')] > self.dEtaJJ)
        else : 
            full_mask = full_mask & (jet_kin[:,self.jet_kin_names.index('DeltaEtaJJ')] < self.dEtaJJ)

        #Apply mask on jet kinematics, truth and pf cands
        jet_kin = jet_kin[full_mask]
        truth = truth[full_mask]
        jet_const[0] = jet_const[0][full_mask]
        jet_const[1] = jet_const[1][full_mask]
        ###############  
        pf_out_list = []
        jet_prop_list = []

        for i_j in range(2): 
            pf_xyze = jet_const[i_j]
            pf_ptep = self.xyze_to_ptep(pf_xyze)
            n_particles = np.sum(pf_xyze[:,:,self.pf_kin_names.index('E')]!=0,axis=1) #E is 3rd
            #this is fancy way of getting only number of particles that are present 
            #pf_xyze_out = list(map(get_present_constit,pf_xyze,n_particles))
            #pf_ptep_out = list(map(get_present_constit,pf_ptep,n_particles))
            #pf_tot_out = list(map(concat_features,pf_xyze_out,pf_ptep_out))
            pf_tot_out = np.dstack([pf_xyze,pf_ptep]) #horizonal stack
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
        self.set_up_scaler()
        #interweavign lists to have subleading jet always follow leading
        pf_cands = np.stack(list(itertools.chain(*zip(pf_out_list[0] , pf_out_list[1]))),axis=0)
        jet_prop = np.stack(list(itertools.chain(*zip(jet_prop_list[0] , jet_prop_list[1]))),axis=0)
        event_idx = k*self.chunk_size

        if write_files :
            with h5py.File(osp.join(self.processed_dir, out_file.format(event_idx)), 'w') as outFile:
                outFile.create_dataset('tot_n_jets', data=[jet_prop.shape[0]], compression='gzip')
                outFile.create_dataset('pf_cands', data=np.array(pf_cands), compression='gzip')
                outFile.create_dataset('jet_props', data=np.array(jet_prop), compression='gzip')
        else:
            return np.array(pf_cands),np.array(jet_prop)


    def process(self):
        """
        Split processing of dataset across multiple processes.
        """
        
        for raw_path in self.raw_paths: #loop over raw files
            #pars = []
            for k in range(self.n_proc):
                # to do it sequentially
                self.process_one_chunk(raw_path, k)

                # to do it with multiprocessing
                #pars += [(self, raw_path, k)]
           # pool = multiprocessing.Pool(self.n_proc)
           # pool.map(process_func, pars)

    def process_on_fly(self):
        ''' Implementation is not finalized, decide whether generator or reader is needed'''
        pf_cands_concat = []
        jet_prop_concat = []
        for raw_path in self.raw_paths: #loop over raw files
            #pars = []
            for k in range(self.n_proc):
                # to do it sequentially
                pf_cands_concat, jet_prop_concat = self.process_one_chunk(raw_path, k,write_files=False)



    def calculate_offsets(self):
        for path in self.processed_paths:
            with h5py.File(path, 'r') as f:
                self.strides.append(f['tot_n_jets'][()][0])
        self.strides = np.cumsum(self.strides)


    def get_pfcands_jet_prop(self, n_evt):
        """ Used by PyTorch DataSet class """    
        file_idx = 0 #takes first file only
        if n_evt > self.strides[file_idx+1]:
            n_evt = self.strides[file_idx+1]
        with h5py.File(self.processed_paths[file_idx],'r') as f:
            n_particles = f['jet_props'][0:n_evt,0].astype(int)
            pf_cands = np.array(f['pf_cands'][0:n_evt,:,:])
            jet_prop = np.array(f['jet_props'][0:n_evt,:])
            if self.scaler is not None :
                pf_cands[:,:,self.idx_gev]/=self.scaler.std_gev
                pf_cands[:,:,self.idx_coord]/=self.scaler.std_coord
            pf_cands = list(map(get_present_constit,pf_cands,n_particles))
        return pf_cands, jet_prop 


    def in_memory_data(self,n_evt):
        pf_cands, jet_prop = self.get_pfcands_jet_prop(n_evt)
        datas = []
        n_jets = len(pf_cands)
        n_particles = [pf_cands[i_evt].shape[0] for i_evt in range(n_jets)]
        adj = [csr_matrix(np.ones((n_part,n_part)) - np.eye(n_part))  for n_part in n_particles]
        edge_index = [from_scipy_sparse_matrix(a)[0] for a in adj]        
        x = [torch.tensor(pf_cands[i_evt], dtype=torch.float) for i_evt in range(n_jets)]
        u = [torch.tensor(jet_prop[i_evt], dtype=torch.float) for i_evt in range(n_jets)]
        datas = [Data(x=x_jet, edge_index=edge_index_jet,u=torch.unsqueeze(u_jet, 0)) for x_jet,edge_index_jet,u_jet in zip(x,edge_index,u)]
        return datas


    def get(self, idx):
        """ Used by PyTorch DataSet class """    
        file_idx = np.searchsorted(self.strides, idx) - 1
        idx_in_file = idx - self.strides[max(0, file_idx)] - 1
        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        #with h5py.File(self.processed_paths[file_idx],'r') as f:
        if self.current_file_idx != file_idx:
            self.current_file_idx = file_idx
            self.current_in_file = h5py.File(self.processed_paths[self.current_file_idx],'r')
        n_particles = int(self.current_in_file['jet_props'][idx_in_file,0])
        adj = csr_matrix(np.ones((n_particles,n_particles)) - np.eye(n_particles)) 
        edge_index,_ = from_scipy_sparse_matrix(adj)
        pf_cands = np.array(self.current_in_file['pf_cands'][idx_in_file,:n_particles,:])
        if self.scaler is not None :
            pf_cands[:,self.idx_gev]/=self.scaler.std_gev
            pf_cands[:,self.idx_coord]/=self.scaler.std_coord
        x = torch.tensor(pf_cands, dtype=torch.float)
        u = torch.tensor(self.current_in_file['jet_props'][idx_in_file], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index,u=torch.unsqueeze(u, 0))
        return data 

    def get_testing(self, idx):
        """ Used by PyTorch DataSet class """
        tot_num_files = self.n_events // self.n_events_merge #+ 1 if (self.n_events % self.n_events_merge) 
        for i_f in  tot_num_files:       
            p = osp.join(self.processed_dir, self.processed_file_names[i_f])
            data = torch.load(p)
            yield data[idx]
        #return data
 
    def _process(self):
        """
        Checks if we want to process the raw file into a dataset. If files 
        already present skips processing.
        """

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        print('Processing...')

        Path(self.processed_dir).mkdir(exist_ok=True)
        self.process()


        print('Done!')


        
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
        

