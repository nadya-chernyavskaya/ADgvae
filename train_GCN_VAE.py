import os
import numpy as np
from collections import namedtuple
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
from importlib import reload
import tensorflow as tf
print('tensorflow version: ', tf.__version__)
import setGPU

import models.models as models
import models.PNmodel as pn
import models.losses as losses
import utils.preprocessing as prepr

# ********************************************************
#       runtime params
# ********************************************************

Parameters = namedtuple('Parameters', 'model latent_dim beta_kl kl_warmup_time epochs train_total_n valid_total_n batch_n activation learning_rate')
params = Parameters(model='PN_AE',
                    latent_dim=30, 
                    beta_kl=50, 
                    kl_warmup_time=0, 
                    epochs=20, 
                    train_total_n=int(1*10e5), 
                    valid_total_n=int(5*10e3), 
                    batch_n=256, 
                    activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                    learning_rate=0.001)

# ********************************************************
#       prepare training and validation data
# ********************************************************

TRAIN_PATH = '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_sideband_parts/'
filename_bg = TRAIN_PATH + 'qcd_sqrtshatTeV_13TeV_PU40_NEW_sideband_000.h5'
batch_size = params.batch_n
train_set_size = int((params.train_total_n//batch_size) * batch_size)
nodes_n, feat_sz, particles_bg, A_bg, A_tilde_bg = prepr.prepare_data(filename_bg,train_set_size,0,train_set_size+1)

VALID_PATH = '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/'
filename_bg_valid = VALID_PATH + 'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_000.h5'
valid_set_size = int((params.valid_total_n//batch_size) * batch_size)
_,_, particles_bg_valid, A_bg_valid, A_tilde_bg_valid = prepr.prepare_data(filename_bg_valid,valid_set_size,0,valid_set_size+1)

# *******************************************************
#                       training options
# *******************************************************

optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

# *******************************************************
#                       logging and callbacks
# *******************************************************
timestamp = str(datetime.now().isoformat(timespec='minutes').replace(':',"_").replace('T','_T_').replace('-','_'))
checkpoint_filepath = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/output_models/{}_weights_'.format(params.model)+timestamp+'.{epoch:02d}-{val_loss:.3f}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, verbose=2),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2),
          #  models.KLWarmupCallback(), #only for VAE
            model_checkpoint_callback] 


# *******************************************************
#                       build model, train and save
# *******************************************************

#gcnvae = models.GCNVariationalAutoEncoder(nodes_n=nodes_n, feat_sz=feat_sz, activation=params.activation,latent_dim=params.latent_dim,beta_kl=params.beta_kl,kl_warmup_time=params.kl_warmup_time)
#gcnvae.compile(optimizer=optimizer)
#print('>>> Launching Training')
#gcnvae.fit(particles_bg, A_bg, epochs=params.epochs, batch_size=batch_size, validation_data = ((particles_bg_valid, A_bg_valid)), callbacks=callbacks) 
#gcnvae.save('output_model_saved_GCN_VAE_{}'.format(timestamp))

#Particle Net
input_shapes = {}
input_shapes['points'] =  [nodes_n,feat_sz-1] #using only coordinates eta phi
input_shapes['features'] = [nodes_n,feat_sz]
input_shapes['mask'] = None
pnae = pn.get_particle_net_lite_ae(input_shapes)
pnae.summary()
pnae.compile(optimizer=optimizer, loss=losses.threeD_loss)
history = pnae.fit((particles_bg[:,:,0:2], particles_bg) , particles_bg,
                    validation_data = ((particles_bg_valid[:,:,0:2], particles_bg_valid) , particles_bg_valid),
                    epochs=10, 
                    batch_size=128, 
                    verbose=1,
                    callbacks=callbacks) 
pnae.save('output_model_saved_{}_{}'.format(params.model,timestamp))


