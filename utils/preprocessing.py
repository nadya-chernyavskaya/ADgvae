import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plti
import sklearn.utils as skutil

def log_transform(x):
	return np.where(x==0,-10,np.log(x))

def transform_min_max(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def transform_mean_std(x):
    return (x-np.mean(x))/(3*np.std(x))


def mask_training_cuts(constituents, features):
    ''' get mask for training cuts requiring a jet-pt > 200'''
    jetPt_cut = 200.
    idx_j1Pt, idx_j2Pt = 1, 6
    mask_j1 = features[:, idx_j1Pt] > jetPt_cut
    mask_j2 = features[:, idx_j2Pt] > jetPt_cut
    ''' normalize jet constituents pt to the jet pt'''
    constituents[:,0,:,2] = np.where(features[:, idx_j1Pt,None]!=0, constituents[:,0,:,2]/features[:, idx_j1Pt,None],0.) #pt is 2nd
    constituents[:,1,:,2] = np.where(features[:, idx_j2Pt,None]!=0, constituents[:,1,:,2]/features[:, idx_j2Pt,None],0.) #pt is 2nd
    ''' log transform pt of constituents'''
    constituents[:,0,:,2] = log_transform(constituents[:,0,:,2]) 
    constituents[:,1,:,2] = log_transform(constituents[:,1,:,2]) 
    return mask_j1, mask_j2

def constituents_to_input_samples(constituents, mask_j1, mask_j2): # -> np.ndarray
        const_j1 = constituents[:,0,:,:][mask_j1]
        const_j2 = constituents[:,1,:,:][mask_j2]
        samples = np.vstack([const_j1, const_j2])
        np.random.shuffle(samples) #this will only shuffle jets
        samples = np.array([skutil.shuffle(item) for item in samples]) #this is pretty slow though
        return samples  

def events_to_input_samples(constituents, features):
    mask_j1, mask_j2 = mask_training_cuts(constituents, features)
    return constituents_to_input_samples(constituents, mask_j1, mask_j2)


def normalize_features(particles):
    idx_eta, idx_phi, idx_pt = range(3)
    # min-max normalize pt
    particles[:,:,idx_pt] = transform_min_max(particles[:,:,idx_pt]) 
    # standard normalize angles
    particles[:,:,idx_eta] = transform_mean_std(particles[:,:,idx_eta]) 
    particles[:,:,idx_phi] = transform_mean_std(particles[:,:,idx_phi]) 
    return particles


def normalized_adjacency(A):
    D = np.array(np.sum(A, axis=2), dtype=np.float32) # compute outdegree (= rowsum)
    D = np.nan_to_num(np.power(D,-0.5), posinf=0, neginf=0) # normalize (**-(1/2))
    D = np.asarray([np.diagflat(dd) for dd in D]) # and diagonalize
    return np.matmul(D, np.matmul(A, D))

def make_adjacencies(particles):
    real_p_mask = particles[:,:,0] > 0 # construct mask for real particles
    adjacencies = (real_p_mask[:,:,np.newaxis] * real_p_mask[:,np.newaxis,:]).astype('float32')
    return adjacencies


def prepare_data(filename,num_instances,start=0,end=-1):
    # set the correct background filename
    filename = filename
    data = h5py.File(filename, 'r') 
    constituents = data['jetConstituentsList'][start:end,]
    features = data['eventFeatures'][start:end,]
    constituents = constituents[:,:,0:50,:] #first select some, as they are ordered in pt, and we shuffle later
    samples = events_to_input_samples(constituents, features)
    # The dataset is N_jets x N_constituents x N_features
    njet     = samples.shape[0]
    if (njet > num_instances) : samples = samples[:num_instances,:,:]
    nodes_n = samples.shape[1]
    feat_sz    = samples.shape[2]
    print('Number of jets =',njet)
    print('Number of constituents (nodes) =',nodes_n)
    print('Number of features =',feat_sz)
    A = make_adjacencies(samples)
    A_tilde = normalized_adjacency(A)
    samples = normalize_features(samples)
    return nodes_n, feat_sz, samples, A, A_tilde

def prepare_data_constituents(filename,num_instances,start=0,end=-1):
    # set the correct background filename
    filename = filename
    data = h5py.File(filename, 'r') 
    constituents = data['jetConstituentsList'][start:end,]
    features = data['eventFeatures'][start:end,]
    constituents = constituents[:,:,0:50,:] #first select some, as they are ordered in pt, and we shuffle later
    samples = events_to_input_samples(constituents, features)
    # The dataset is N_jets x N_constituents x N_features
    njet     = samples.shape[0]
    if (njet > num_instances) : samples = samples[:num_instances,:,:]
    nodes_n = samples.shape[1]
    feat_sz    = samples.shape[2]
    print('Number of jets =',njet)
    print('Number of constituents (nodes) =',nodes_n)
    print('Number of features =',feat_sz)
    samples = normalize_features(samples)
    return nodes_n, feat_sz, samples
