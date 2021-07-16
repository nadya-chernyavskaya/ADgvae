import os
import numpy as np
from collections import namedtuple
import h5py
import matplotlib.pyplot as plt
from importlib import reload
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
print('tensorflow version: ', tf.__version__)

import models.models as models
import utils.preprocessing as prepr

# ********************************************************
#       runtime params
# ********************************************************

Parameters = namedtuple('Parameters', 'input_shape latent_dim kl_warmup_time epochs train_total_n valid_total_n batch_n activation learning_rate')
params = Parameters(input_shape=(100,3),
                    latent_dim=30, 
                    kl_warmup_time=10, 
                    epochs=300, 
                    train_total_n=int(10e4), 
                    valid_total_n=int(10e3), 
                    batch_n=256, 
                    activation=tf.nn.tanh,
                    learning_rate=0.001)

# ********************************************************
#       prepare training and validation data
# ********************************************************

TRAIN_PATH = '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_sideband_parts/'
filename_bg = TRAIN_PATH + 'qcd_sqrtshatTeV_13TeV_PU40_NEW_sideband_000.h5'
batch_size = params.batch_n
train_set_size = (params.train_total_n//batch_size) * batch_size
nodes_n, feat_sz, particles_bg, A_bg, A_tilde_bg = prepr.prepare_data(filename_bg,train_set_size,0,train_set_size+1)

VALID_PATH = '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts'
filename_bg_valid = TRAIN_PATH + 'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_000.h5'
valid_set_size = (params.valid_total_n//batch_size) * batch_size
_,_, particles_bg_valid, A_bg_valid, A_tilde_bg_valid = prepr.prepare_data(filename_bg,valid_set_size,0,valid_set_size+1)

# *******************************************************
#                       training options
# *******************************************************

optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

# *******************************************************
#                       build model
# *******************************************************

gcnvae = models.GCNVariationalAutoEncoder(nodes_n=params.input_shape[0], feat_sz=params.input_shape[1], activation=params.activation,latent_dim=params.latent_dim,kl_warmup_time=params.kl_warmup_time)
gcnvae.compile(optimizer=optimizer, run_eagerly=True)

callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, verbose=2,min_lr=0.00001),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2),
            models.KLWarmupCallback()] 

# *******************************************************
#                       train and save
# *******************************************************
print('>>> Launching Training')
gcnvae.fit(particles_bg, A_bg, epochs=params.epochs, batch_size=batch_size, validation_data = ((particles_bg_valid, A_bg_valid)), callbacks=callbacks) 

gcnvae.save('output_model_saved_003')
