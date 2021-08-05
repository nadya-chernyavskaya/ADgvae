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
import models.ParticleNetAE as pnae
import models.losses as losses
import utils.preprocessing as prepr

# ********************************************************
#       runtime params
# ********************************************************

Parameters = namedtuple('Parameters', 'model latent_dim beta_kl kl_warmup_time epochs train_total_n valid_total_n batch_n activation learning_rate')
params = Parameters(model='PN_AE',
                    latent_dim=10, 
                    beta_kl=10, 
                    kl_warmup_time=3, 
                    epochs=100, 
                    train_total_n=int(1*10e6), 
                    valid_total_n=int(1*10e4), 
                    batch_n=128, 
                    activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                    learning_rate=0.01)

# ********************************************************
#       prepare training and validation data
# ********************************************************

DATA_PATH = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/input/'
filename_bg = DATA_PATH + 'QCD_training_data_100const_03_08_2021.h5'
inFile = h5py.File(filename_bg, 'r')
#particles_bg = inFile['particle_bg'][()]
#particles_bg_valid = inFile['particle_bg_valid'][()]
particles_bg = inFile['particle_bg'][0:params.train_total_n]
particles_bg_valid = inFile['particle_bg_valid'][0:params.valid_total_n]
print('Training/validation on {}/{} samples'.format(particles_bg.shape[0],particles_bg_valid.shape[0]))
nodes_n = particles_bg.shape[1]
feat_sz = particles_bg.shape[2]
batch_size = params.batch_n

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

callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,min_delta=0.0005, patience=5, verbose=2),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2),
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
class _DotDict:
    pass

setting = _DotDict()
 # conv_params: list of tuple in the format (K, (C1, C2, C3))
setting.conv_params = [
   (15, ([20,20,20])),
 #   (7, (32, 32, 32)),
 #   (7, (64, 64, 64)),
        ]
setting.conv_params_encoder_input = 20 #20
setting.conv_params_encoder = []
setting.conv_params_decoder = [10]  
setting.with_bn = True
# conv_pooling: 'average' or 'max'
setting.conv_pooling = 'average'
setting.conv_linking = 'sum' #concat or sum # right now this is off
setting.num_points = nodes_n #num of original consituents
setting.num_features = feat_sz #num of original features
setting.input_shapes = {'points': [nodes_n,feat_sz-1],'features':[nodes_n,feat_sz]}
setting.latent_dim = params.latent_dim
setting.ae_type = params.model  #ae or vae 
setting.beta_kl = 10
setting.kl_warmup_time = params.kl_warmup_time
setting.activation = params.activation

model = pnae.PNVAE(setting=setting,name=params.model)
model.compile(optimizer=optimizer)
_,_= model.predict([particles_bg[0:2,:,0:2],particles_bg[0:2,:,:]]) #hack to save the model
model.save('output_model_saved_{}_{}'.format(params.model,timestamp))

history = model.fit((particles_bg[:,:,0:2], particles_bg) , particles_bg,
                    validation_data = ((particles_bg_valid[:,:,0:2], particles_bg_valid) , particles_bg_valid),
                    epochs=params.epochs, 
                    batch_size=batch_size, 
                    verbose=1,
                    callbacks=callbacks) 


