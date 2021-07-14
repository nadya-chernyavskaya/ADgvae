import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

def normalize_features(particles):
    idx_pt, idx_eta, idx_phi, idx_class = range(4)
    # min-max normalize pt
    particles[:,:,idx_pt] = (particles[:,:,idx_pt] - np.min(particles[:,:,idx_pt])) / (np.max(particles[:,:,idx_pt])-np.min(particles[:,:,idx_pt]))
    # standard normalize angles
    particles[:,:,idx_eta] = (particles[:,:,idx_eta] - np.mean(particles[:,:,idx_eta]))/np.std(particles[:,:,idx_eta])
    particles[:,:,idx_phi] = (particles[:,:,idx_phi] - np.mean(particles[:,:,idx_phi]))/np.std(particles[:,:,idx_phi])
    # min-max normalize class label
    particles[:,:,idx_class] = (particles[:,:,idx_class] - np.min(particles[:,:,idx_class])) / (np.max(particles[:,:,idx_class])-np.min(particles[:,:,idx_class]))
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


def prepare_data(filename,start=0,end=-1):
    # set the correct background filename
    filename = filename
    ff = h5py.File(filename, 'r')
    particles = np.asarray(ff.get('Particles'))
    nodes_n = particles.shape[1]
    feat_sz = particles.shape[2]
    particles = particles[start:end]
    A = make_adjacencies(particles)
    A_tilde = normalized_adjacency(A)
    particles = normalize_features(particles)
    return nodes_n, feat_sz, particles, A, A_tilde
