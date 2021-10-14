import numpy as np
import h5py
import os
from importlib import reload
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils.preprocessing as prepr


#Data Samples
DATA_PATH = '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/'

TRAIN_NAME = 'qcd_sqrtshatTeV_13TeV_PU40_NEW_sideband'
filename_bg = DATA_PATH + TRAIN_NAME + '_parts/' + TRAIN_NAME + '_000.h5'
batch_size = 128
train_set_size = int((1*10e6//batch_size)*batch_size) #10 million probably good

nodes_n, feat_sz, particles_bg  = prepr.prepare_data_constituents(filename_bg,train_set_size,0,train_set_size+1)


# BG validation
VALID_NAME = 'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband'
filename_bg_valid = DATA_PATH + VALID_NAME + '_parts/' + VALID_NAME + '_000.h5'
valid_set_size = int((5*10e4//batch_size)*batch_size)
_,_, particles_bg_valid = prepr.prepare_data_constituents(filename_bg_valid,valid_set_size,0,valid_set_size+1)


#BG test
_,_, particles_bg_test = prepr.prepare_data_constituents(filename_bg_valid,5000,valid_set_size+1,valid_set_size+5000)


output_file = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/input/QCD_training_data_100const_05_08_2021.h5'
with h5py.File(output_file, 'w')as outFile:
    outFile.create_dataset('particle_bg', data=particles_bg, compression='gzip')
    outFile.create_dataset('particle_bg_valid', data=particles_bg_valid, compression='gzip')
    outFile.create_dataset('particle_bg_test', data=particles_bg_test, compression='gzip')

