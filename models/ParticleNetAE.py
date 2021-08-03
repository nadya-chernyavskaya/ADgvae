import tensorflow as tf
import tensorflow.keras.layers as klayers
from tensorflow import keras
import models.losses as losses
import models.layers as layers
from keras import backend as K
#import models.PNmodel as pn
import models.custom_functions as funcs


class PNVAE(tf.keras.Model):

   def __init__(self,setting, **kwargs):
      super(PNVAE, self).__init__(**kwargs)
      self.setting = setting
      self.ae_input_dim = setting.conv_params[-1][-1][-1]*2 if setting.conv_linking == 'concat' else setting.conv_params[-1][-1][-1]
      self.with_bn = True 
      self.latent_dim = setting.latent_dim
      self.activation = setting.activation
      self.kl_warmup_time = setting.kl_warmup_time
      self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)
      self.particlenet = self.build_particlenet()
      self.sampling = self.build_sampling()
      self.encoder = self.build_encoder()
      self.decoder = self.build_decoder()

      self.loss_tracker = keras.metrics.Mean(name="loss")
      self.reco_loss_tracker = keras.metrics.Mean(name="reco_loss")
      self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


   def build_edgeconv(self,points,features,K=7,channels=32,name=''):
      """EdgeConv
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """
      with tf.name_scope('EdgeConv_'):        
         # distance
         D = funcs.batch_distance_matrix_general(points, points)  # (N, P, P)
         _, indices = tf.nn.top_k(-D, k=K + 1)  # (N, P, K+1)
         indices = indices[:, :, 1:]  # (N, P, K)

         fts = features
         knn_fts = funcs.knn(self.setting.num_points, K, indices, fts)  # (N, P, K, C)
         knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, K, 1))  # (N, P, K, C)
         knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

         x = knn_fts
         for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                        use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (name, idx))(x)
            if self.with_bn:
               x = keras.layers.BatchNormalization(name='%s_bn%d' % (name, idx))(x)
            if self.activation:
               x = keras.layers.Activation(self.activation, name='%s_act%d' % (name, idx))(x)

         if self.setting.conv_pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
         else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')
                
         # shortcut of constituents features
         sc = keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                     use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % name)(tf.expand_dims(features, axis=2))
         if self.with_bn:
                sc = keras.layers.BatchNormalization(name='%s_sc_bn' % name)(sc)
         sc = tf.squeeze(sc, axis=2)

         x = sc + fts #sum by default, original PN
         if self.setting.conv_linking == 'concat': #concat or sum
            x = tf.concat([sc,fts],axis=2) 
         if self.activation:
            x =  keras.layers.Activation(self.activation, name='%s_sc_act' % name)(x)  # (N, P, C') #TO DO : try with concatenation instead of sum
         return x



   def build_particlenet(self):
        with tf.name_scope('ParticleNetBase'):

           points = klayers.Input(name='points', shape=self.setting.input_shapes['points'])
           features = klayers.Input(name='features', shape=self.setting.input_shapes['features']) if 'features' in self.setting.input_shapes else None

           #mask = keras.Input(name='mask', shape=self.setting.input_shapes['mask']) if 'mask' in self.setting.input_shapes else None
           mask = None #TO DO : need to check how to implement that when/if we need it

           if mask is not None:
               mask = tf.cast(tf.not_equal(mask, 0), dtype='float32')  # 1 if valid
               coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype='float32'))  # make non-valid positions to 99

           fts = tf.squeeze(klayers.BatchNormalization(name='%s_fts_bn' % self.name)(tf.expand_dims(features, axis=2)), axis=2)
           for layer_idx, layer_param in enumerate(self.setting.conv_params):
               K, channels = layer_param
               if mask is not None:
                   pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
               else : pts=points
               fts_shape = fts.get_shape().as_list()
               pts_shape = pts.get_shape().as_list()
               fts = self.build_edgeconv(pts,fts,K=K,channels=channels,name='%s_%i'%(self.name,layer_idx))

           if mask is not None:
               fts = tf.multiply(fts, mask)

           pool = tf.reduce_mean(fts, axis=1)  # (N, C)  #pooling over all jet constituents

           particle_net_base = tf.keras.Model(inputs=(points,features), outputs=pool,name='ParticleNetBase')
           particle_net_base.summary()
           return particle_net_base 

   def build_sampling(self):
        input_layer   = klayers.Input(shape=(self.ae_input_dim, ), name='sampling_input')
        z_mean = keras.layers.Dense(self.setting.latent_dim, name = 'z_mean', activation=self.activation,kernel_initializer='glorot_normal' )(input_layer)
        z_log_var = keras.layers.Dense(self.setting.latent_dim, name = 'z_log_var', activation=self.activation,kernel_initializer='glorot_normal' )(input_layer)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) #,mean=0., stddev=0.1
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        sampling_model = tf.keras.Model(inputs=(input_layer), outputs=[z,z_mean,z_log_var],name='SamplingLayer')
        return sampling_model  

   def build_encoder(self):
        input_layer   = klayers.Input(shape=(self.ae_input_dim, ), name='encoder_input')
        if 'vae'.lower() in self.setting.ae_type :
            encoder_output = self.sampling(input_layer)
            encoder_model = tf.keras.Model(inputs=(input_layer), outputs=encoder_output,name='Encoder')
        else :  
            latent_space = keras.layers.Dense(self.setting.latent_dim,activation=self.activation,
                                              kernel_initializer='glorot_normal')(input_layer)
            encoder_output = [latent_space]
            encoder_model = tf.keras.Model(inputs=(input_layer), outputs=encoder_output,name='Encoder')
        encoder_model.summary() 
        return encoder_model


   def build_decoder(self):
        input_layer   = klayers.Input(shape=(self.setting.latent_dim, ), name='decoder_input')
        num_dense_channels = self.setting.conv_params_decoder[0]

        #x = klayers.Dense((self.setting.num_points*num_dense_channels),activation=self.activation )(input_layer)
        #x = klayers.BatchNormalization(name='%s_dense_0' % (self.name))(x)
        x = keras.layers.Dense((self.setting.num_points*num_dense_channels),
                               kernel_initializer='glorot_normal')(input_layer) 
        #TO DO: order of BN->Activation or the other way around can have impact, check
        if self.with_bn:
            x = klayers.BatchNormalization(name='%s_dense_0' % (self.name))(x)
        if self.activation:
            x = klayers.Activation(self.activation, name='%s_act_0' % (self.name))(x)  
        x = klayers.Reshape((self.setting.num_points,num_dense_channels), input_shape=(self.setting.num_points*num_dense_channels,))(x)
 
        for layer_idx in range(1,len(self.setting.conv_params_decoder)):
            layer_param  = self.setting.conv_params_decoder[layer_idx]
            #1D and 2D  Conv layers with kernel and stride side of 1 are identical operations, but for 2D first need to expand then to squeeze
            #x = tf.squeeze(keras.layers.Conv2D(layer_param, kernel_size=(1, 1), strides=1, data_format='channels_last',
            #                        use_bias=False if self.with_bn else True, activation=self.activation, kernel_initializer='glorot_normal',
            #                        name='%s_conv_%d' % (self.name,layer_idx))(tf.expand_dims(x, axis=2)),axis=2)  
            #x = klayers.BatchNormalization(name='%s_bn_%d' % (self.name,layer_idx))(x)
            x = klayers.Conv2D(layer_param, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal',
                                    name='%s_conv_%d' % (self.name,layer_idx))(tf.expand_dims(x, axis=2))
            if self.with_bn:
                x = klayers.BatchNormalization(name='%s_bn_%d' % (self.name,layer_idx))(x)
            x = tf.squeeze(x, axis=2)
            if self.activation:
                x = klayers.Activation(self.activation, name='%s_act_%d' % (self.name,layer_idx))(x)  

        decoder_output = tf.squeeze(klayers.Conv2D(self.setting.num_features, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=True, activation=self.activation, kernel_initializer='glorot_normal',
                                    name='%s_conv_out' % self.name)(tf.expand_dims(x, axis=2)),axis=2) 
        decoder = tf.keras.Model(inputs=input_layer, outputs=decoder_output,name='Decoder')
        decoder.summary()
        return decoder 


   def call(self, inputs):
        pool_layer = self.particlenet(inputs)
        encoder_output = self.encoder(pool_layer)
        if 'vae'.lower() in self.setting.ae_type :
            z, z_mean, z_log_var = encoder_output
        else : z = encoder_output
        decoder_output = self.decoder(z) 
        return encoder_output, decoder_output

   @property
   def metrics(self):
       return [
           self.loss_tracker,
           self.reco_loss_tracker,
           self.kl_loss_tracker,
       ]

   def train_step(self, data):
        (coord_in, feats_in) , feats_in = data

        with tf.GradientTape() as tape:
            encoder_output, decoder_output  = self((coord_in, feats_in))  # Forward pass
            feats_out = decoder_output
            if 'vae'.lower() in self.setting.ae_type :
                z, z_mean, z_log_var = encoder_output
                loss_reco = tf.math.reduce_mean(losses.threeD_loss(feats_in,feats_out))
                loss_latent = tf.math.reduce_mean(losses.kl_loss(z_mean, z_log_var))
                loss = loss_reco + self.setting.beta_kl  * loss_latent *tf.cond(tf.greater(self.beta_kl_warmup, 0), lambda: self.beta_kl_warmup, lambda: 1.)
            else : 
                z = encoder_output
                loss = tf.math.reduce_mean(losses.threeD_loss(feats_in,feats_out))

      #  metrics = {'loss':loss}
      #  if 'vae'.lower() in self.setting.ae_type :
      #     metrics['loss_reco'] = loss_reco
      #     metrics['loss_latent'] = loss_latent
       
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return_metrics = {
            "loss": self.loss_tracker.result(),
            }
        if 'vae'.lower() in self.setting.ae_type :
            self.reco_loss_tracker.update_state(loss_reco)
            self.kl_loss_tracker.update_state(loss_latent)
            return_metrics["reco_loss"] =  self.reco_loss_tracker.result()
            return_metrics["kl_loss"] =  self.kl_loss_tracker.result()
        return return_metrics


   def test_step(self, data):
        (coord_in, feats_in) , feats_in = data
        encoder_output, decoder_output = self((coord_in, feats_in), training=False)  # Forward pass
        feats_out = decoder_output
        if 'vae'.lower() in self.setting.ae_type :
            z, z_mean, z_log_var = encoder_output
            loss_reco = tf.math.reduce_mean(losses.threeD_loss(feats_in,feats_out))
            loss_latent = tf.math.reduce_mean(losses.kl_loss(z_mean, z_log_var))
            loss = loss_reco + self.setting.beta_kl  * loss_latent *tf.cond(tf.greater(self.beta_kl_warmup, 0), lambda: self.beta_kl_warmup, lambda: 1.)
        else : 
           z = encoder_output
           loss = tf.math.reduce_mean(losses.threeD_loss(feats_in,feats_out))

        metrics = {'loss':loss}
        if 'vae'.lower() in self.setting.ae_type :
           metrics['loss_reco'] = loss_reco
           metrics['loss_latent'] = loss_latent
    
        return metrics
    
    

