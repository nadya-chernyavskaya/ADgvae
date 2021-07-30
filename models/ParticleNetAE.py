import tensorflow as tf
import tensorflow.keras.layers as klayers
from tensorflow import keras
import models.losses as losses
import models.layers as layers
from keras import backend as K
#import models.PNmodel as pn
import models.custom_functions as funcs


class PNVAE(tf.keras.Model):

   def __init__(self,input_shapes,mask,setting, **kwargs):
      super(PNVAE, self).__init__(**kwargs)
      self.input_shapes = input_shapes
      self.setting = setting
      self.latent_dim = setting.latent_dim
      self.activation =klayers.LeakyReLU(alpha=0.1) #TO DO : pass activation
      #  self.kl_warmup_time = setting.kl_warmup_time
      #  self.beta_kl = setting.beta_kl 
      #  self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)
      self.particlenet = self.build_particlenet()
    #  self.encoder = self.build_encoder()
    #  self.decoder = self.build_decoder()
      self.encoder = self._encoder()
      self.decoder = self._decoder()


   def build_edgeconv(self,points,features, input_shape=[0,0],K=7,channels=32):
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
    #  with tf.name_scope('EdgeConv_'):        
      if 1>0:        
      #   points = klayers.Input(name='edgeconv_points', shape=input_shape[0][1:])
      #   features = klayers.Input(name='edgeconv_features', shape=input_shape[1][1:])
         with_bn = True
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
                                        use_bias=False if with_bn else True, kernel_initializer='glorot_normal')(x) #, name='%s_conv%d' % (self.name, idx))(x)
            if with_bn:
               x = keras.layers.BatchNormalization()(x) #(name='%s_bn%d' % (self.name, idx))(x)
            if self.activation:
               x = keras.layers.Activation(self.activation ) (x) #), name='%s_act%d' % (self.name, idx))(x)

         if self.setting.conv_pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
         else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')
                
         # shortcut of constituents features
         sc = keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                     use_bias=False if with_bn else True, kernel_initializer='glorot_normal')(tf.expand_dims(features, axis=2)) #, name='%s_sc_conv' % self.name)(tf.expand_dims(features, axis=2))
         if with_bn:
                sc = keras.layers.BatchNormalization()(sc) #(name='%s_sc_bn' % self.name)(sc)
         sc = tf.squeeze(sc, axis=2)

         if self.activation:
            return keras.layers.Activation(self.activation)(sc+fts) #, name='%s_sc_act' % self.name)(sc + fts)  # (N, P, C') #try with concatenation instead of sum
         else:
            return sc + fts



   def build_particlenet(self):
      #  with tf.name_scope('ParticleNetBase'):
        if 1>0:        

           points = klayers.Input(name='points', shape=self.input_shapes['points'])
           features = klayers.Input(name='features', shape=self.input_shapes['features']) if 'features' in self.input_shapes else None

           #mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
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
             #  fts = layers.EdgeConvModule(self.setting.num_points, K, channels, with_bn=True, activation=self.activation,
              #                 pooling=self.setting.conv_pooling, name='%s_%s%d' % (self.name, 'EdgeConv', layer_idx))(pts, fts)
               fts_shape = fts.get_shape().as_list()
               pts_shape = pts.get_shape().as_list()
               #fts = self.build_edgeconv(input_shape=[pts_shape,fts_shape],K=K,channels=channels)(pts,fts)
               fts = self.build_edgeconv(pts,fts,input_shape=[pts_shape,fts_shape],K=K,channels=channels)


           if mask is not None:
               fts = tf.multiply(fts, mask)

           pool = tf.reduce_mean(fts, axis=1)  # (N, C)  #pooling over all jet constituents

           particle_net_base = tf.keras.Model(inputs=(points,features), outputs=pool,name='ParticleNetBase')
           particle_net_base.summary()
           return particle_net_base 

   def _encoder(self):
       # if 'vae'.lower() in self.setting.ae_type :
       #     z, z_mean_, z_log_var = _sampling(pool_layer, setting=self.setting, name=name)
       #     encoder_output = [z, z_mean_, z_log_var]
        #else :  
        input_layer   = klayers.Input(shape=(self.setting.conv_params[-1][-1][-1], ), name='encoder_input')
        if 1>0 :  
            latent_space = keras.layers.Dense(self.setting.latent_dim,activation=self.activation )(input_layer)
            encoder_output = [latent_space]
        encoder = tf.keras.Model(inputs=input_layer, outputs=encoder_output,name='Encoder')
        encoder.summary()
        return encoder 


   def _decoder(self):
        input_layer   = klayers.Input(shape=(self.setting.latent_dim, ), name='decoder_input')
        num_channels = self.setting.conv_params[-1][-1][-1]
        if 1>0:
            x = keras.layers.Dense((25*self.setting.num_points),activation=self.activation )(input_layer)
            x = keras.layers.BatchNormalization(name='%s_bn_1' % (self.name))(x)
            x = keras.layers.Reshape((self.setting.num_points,25), input_shape=(num_channels*self.setting.num_points,))(x) 
            #1D and 2D  Conv layers with kernel and stride side of 1 are identical operations, but for 2D first need to expand then to squeeze
            x = tf.squeeze(keras.layers.Conv2D(self.setting.num_features*3, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                        use_bias=True, activation =self.activation, kernel_initializer='glorot_normal',
                                        name='%s_conv_0' % self.name)(tf.expand_dims(x, axis=2)),axis=2)  
            x = keras.layers.BatchNormalization(name='%s_bn_2' % (self.name))(x)
            x = tf.squeeze(keras.layers.Conv2D(self.setting.num_features*2, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                        use_bias=True, activation =self.activation, kernel_initializer='glorot_normal',
                                        name='%s_conv_2' % self.name)(tf.expand_dims(x, axis=2)),axis=2)  
            x = keras.layers.BatchNormalization(name='%s_bn_3' % (self.name))(x)
            decoder_output = tf.squeeze(keras.layers.Conv2D(self.setting.num_features, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                        use_bias=True, activation =self.activation, kernel_initializer='glorot_normal',
                                        name='%s_conv_out' % self.name)(tf.expand_dims(x, axis=2)),axis=2) 
            decoder = tf.keras.Model(inputs=input_layer, outputs=decoder_output,name='Decoder')
            decoder.summary()
            return decoder 


   def build_encoder(self):
        input_layer   = klayers.Input(shape=(self.setting.conv_params[-1][-1][-1], ), name='encoder_input')
        encoder_output = layers.ParticleNetEncoder(setting=self.setting, name=self.name)(input_layer)
        encoder = tf.keras.Model(inputs=input_layer, outputs=encoder_output,name='Encoder')
        encoder.summary()
        return encoder 

   def build_decoder(self):
        input_layer   = klayers.Input(shape=(self.setting.latent_dim, ), name='decoder_input')
        decoder_output = layers.ParticleNetDecoder(setting=self.setting, name=self.name)(input_layer)
        decoder = tf.keras.Model(inputs=input_layer, outputs=decoder_output,name='Decoder')
        decoder.summary()
        return decoder 


   def call(self, inputs):
        pool_layer = self.particlenet(inputs)
        encoder_output = self.encoder(pool_layer)
        decoder_output = self.decoder(encoder_output) #has to be changed for VAE
       # encoder_output = self.encoder(pool_layer)
       # decoder_output = self.decoder(encoder_output[0])
        return encoder_output, decoder_output


   def train_step(self, data):
        (coord_in, feats_in) , feats_in = data

        with tf.GradientTape() as tape:
            encoder_output, decoder_output  = self((coord_in, feats_in))  # Forward pass
            # Compute the loss value 
            #TO DO : write loss for VAE/AE
            if 'vae'.lower() in self.setting.ae_type :
                z, z_mean_, z_log_var = encoder_output
            else : z = encoder_output
            feats_out = decoder_output
            loss = tf.math.reduce_mean(losses.threeD_loss(feats_in,feats_out))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


   def test_step(self, data):
        (coord_in, feats_in) , feats_in = data
        encoder_output, decoder_output = self((coord_in, feats_in), training=False)  # Forward pass
        if 'vae'.lower() in self.setting.ae_type :
            z, z_mean_, z_log_var = encoder_output
        else : z = encoder_output
        feats_out = decoder_output
        #TO DO : write loss for VAE/AE
        loss = tf.math.reduce_mean(losses.threeD_loss(feats_in,feats_out))
    
        return {'loss' : loss}
    
    

