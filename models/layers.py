import tensorflow as tf
import tensorflow.keras.layers as klayers
from tensorflow import keras
import models.custom_functions as funcs
 


class GraphConvolution(tf.keras.layers.Layer):
    
    ''' basic graph convolution layer performing act(AXW1 + XW2 + B), nodes+neigbours and self-loop weights plus bias term '''

    def __init__(self, output_sz, activation=tf.keras.activations.linear, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_sz = output_sz
        self.activation = activation

    def build(self, input_shape):
        self.wgt1 = self.add_weight("weight_1",shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        # self-loop weights
        self.wgt2 = self.add_weight("weight_2",shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        self.bias = self.add_weight("bias",shape=[self.output_sz])
        

    def call(self, inputs, adjacency):
        xw1 = tf.matmul(inputs, self.wgt1)
        xw2 = tf.matmul(inputs, self.wgt2)
        axw1 = tf.matmul(adjacency, xw1)
        axw = axw1 + xw2           # add node and neighbours weighted features (self reccurency)
        layer = tf.nn.bias_add(axw, self.bias) 
        return self.activation(layer)
    

    def get_config(self):
        config = super(GraphConvolution, self).get_config()
        config.update({'output_sz': self.output_sz, 'activation': self.activation})
        return config

class GraphConvolutionRecurBias(tf.keras.layers.Layer):
    
    ''' basic graph convolution layer performing act(AXW1 + XW2 + B), nodes+neigbours and self-loop weights plus bias term '''

    def __init__(self, output_sz, activation=tf.keras.activations.linear, **kwargs):
        super(GraphConvolutionRecurBias, self).__init__(**kwargs)
        self.output_sz = output_sz
        self.activation = activation

    def build(self, input_shape):
        self.wgt1 = self.add_weight("weight_1",shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        # self-loop weights
        self.wgt2 = self.add_weight("weight_2",shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        self.bias = self.add_weight("bias",shape=[self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        

    def call(self, inputs, adjacency):
        xw1 = tf.matmul(inputs, self.wgt1)
        xw2 = tf.matmul(inputs, self.wgt2)
        axw1 = tf.matmul(adjacency, xw1)
        axw = axw1 + xw2           # add node and neighbours weighted features (self reccurency)
        layer = tf.nn.bias_add(axw, self.bias) 
        return self.activation(layer)
    

    def get_config(self):
        config = super(GraphConvolutionRecurBias, self).get_config()
        config.update({'output_sz': self.output_sz, 'activation': self.activation})
        return config


class GraphConvolutionBias(tf.keras.layers.Layer):
    
    ''' basic graph convolution layer performing act(AXW1 + B), nodes+neigbours plus bias term '''

    def __init__(self, output_sz, activation=tf.keras.activations.linear, **kwargs):
        super(GraphConvolutionBias, self).__init__(**kwargs)
        self.output_sz = output_sz
        self.activation = activation

    def build(self, input_shape):
        self.wgt1 = self.add_weight("weight_1",shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        # self-loop weights
        self.bias = self.add_weight("bias",shape=[self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        

    def call(self, inputs, adjacency):
        xw1 = tf.matmul(inputs, self.wgt1)
        axw1 = tf.matmul(adjacency, xw1)
        layer = tf.nn.bias_add(axw1, self.bias) 
        return self.activation(layer)
    

    def get_config(self):
        config = super(GraphConvolutionBias, self).get_config()
        config.update({'output_sz': self.output_sz, 'activation': self.activation})
        return config



class InnerProductDecoder(tf.keras.layers.Layer):

    ''' inner product decoder reconstructing adjacency matrix as act(z^T z) 
        input assumed of shape [batch_sz x n_nodes x z_d]
        where 
            batch_sz can be 1 for single example feeding
            n_nodes ... number of nodes in graph
            z_d ... dimensionality of latent space
    '''

    def __init__(self, activation=tf.keras.activations.linear, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.activation = activation

    def call(self, inputs):
        perm = [0, 2, 1] if len(inputs.shape) == 3 else [1, 0]
        z_t = tf.transpose(inputs, perm=perm)
        adjacency_hat = tf.matmul(inputs, z_t)
        return self.activation(adjacency_hat)

    def get_config(self):
        config = super(InnerProductDecoder, self).get_config()
        return config



class EdgeConvModule(tf.keras.layers.Layer):
    """EdgeConv
    Args:
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
    def __init__(self, num_points, K, channels, with_bn=True, activation=klayers.LeakyReLU(alpha=0.1), pooling='average', name='EdgeConvModule'):
        super(EdgeConvModule, self).__init__(name=name)
        self.num_points = num_points    
        self.K = K    
        self.channels = channels    
        self.with_bn = with_bn    
        self.activation = activation    
        self.pooling = pooling    

    def call(self, points, features):
        # distance
        D = funcs.batch_distance_matrix_general(points, points)  # (N, P, P)
        _, indices = tf.nn.top_k(-D, k=self.K + 1)  # (N, P, K+1)
        indices = indices[:, :, 1:]  # (N, P, K)

        fts = features
        knn_fts = funcs.knn(self.num_points, self.K, indices, fts)  # (N, P, K, C)
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, self.K, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

        x = knn_fts
        for idx, channel in enumerate(self.channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (self.name, idx))(x)
            if self.with_bn:
                x = keras.layers.BatchNormalization(name='%s_bn%d' % (self.name, idx))(x)
            if self.activation:
                x = keras.layers.Activation(self.activation, name='%s_act%d' % (self.name, idx))(x)

        if self.pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')

        # shortcut of constituents features
        sc = keras.layers.Conv2D(self.channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % self.name)(tf.expand_dims(features, axis=2))
        if self.with_bn:
            sc = keras.layers.BatchNormalization(name='%s_sc_bn' % self.name)(sc)
        sc = tf.squeeze(sc, axis=2)

        if self.activation:
            return keras.layers.Activation(self.activation, name='%s_sc_act' % self.name)(sc + fts)  # (N, P, C') #try with concatenation instead of sum
        else:
            return sc + fts

    def get_config(self):
        config = super(EdgeConvModule, self).get_config()
        config.update({'activation': self.activation, 'name': self.name})
        return config 


class ParticleNetBase(tf.keras.layers.Layer):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optinal
    def __init__(self, setting, activation=klayers.LeakyReLU(alpha=0.1), name='ParticleNetBase'):
        super(ParticleNetBase, self).__init__(name=name)
        self.setting = setting    
        self.activation = activation    

    def call(self, points, features=None, mask=None):
        if features is None:
            features = points

        if mask is not None:
            mask = tf.cast(tf.not_equal(mask, 0), dtype='float32')  # 1 if valid
            coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype='float32'))  # make non-valid positions to 99

        fts = tf.squeeze(keras.layers.BatchNormalization(name='%s_fts_bn' % self.name)(tf.expand_dims(features, axis=2)), axis=2)
        for layer_idx, layer_param in enumerate(self.setting.conv_params):
            K, channels = layer_param
            if mask is not None:
                pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
            else : pts=points
            fts = EdgeConvModule(self.setting.num_points, K, channels, with_bn=True, activation=self.activation,
                            pooling=self.setting.conv_pooling, name='%s_%s%d' % (self.name, 'EdgeConv', layer_idx))(pts, fts)

        if mask is not None:
            fts = tf.multiply(fts, mask)

        pool = tf.reduce_mean(fts, axis=1)  # (N, C)  #pooling over all jet constituents
        return pool


    def get_config(self):
        config = super(Sampling, self).get_config()
        config.update({'activation': self.activation, 'name': self.name})
        return config  


class Sampling(tf.keras.layers.Layer):
    def __init__(self, setting, activation=klayers.LeakyReLU(alpha=0.1), name='Sampling'):
        super(Sampling, self).__init__(name=name)
        self.setting = setting    
        self.activation = activation

    def call(self, input_layer):
        #Latent dimension and sampling 
        z_mean = keras.layers.Dense(setting.latent_dim, name = 'z_mean', activation=activation )(input_layer)
        z_log_var = keras.layers.Dense(setting.latent_dim, name = 'z_log_var', activation=activation )(input_layer)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z, z_mean, z_log_var  

    def get_config(self):
        config = super(Sampling, self).get_config()
        config.update({'activation': self.activation})
        return config 




class ParticleNetEncoder(tf.keras.layers.Layer):
    def __init__(self, setting, activation=klayers.LeakyReLU(alpha=0.1), name='PNEncoder'):
        super(ParticleNetEncoder, self).__init__(name=name)
        self.setting = setting    
        self.activation = activation  

    def call(self, input_layer):
        if 'vae'.lower() in setting.ae_type :
            z, z_mean_, z_log_var = Sampling(setting=self.setting, name=self.name)(input_layer)
            encoder_output = [z, z_mean_, z_log_var]
        else :  
            latent_space = klayers.Dense(setting.latent_dim,activation=self.activation )(input_layer)
            encoder_output = [latent_space]
        return encoder_output

    def get_config(self):
        config = super(ParticleNetEncoder, self).get_config()
        config.update({'activation': self.activation, 'name': self.name})
        return config 



class ParticleNetDecoder(tf.keras.layers.Layer):
    def __init__(self, setting, activation=klayers.LeakyReLU(alpha=0.1), name='PNDecoder'):
        super(ParticleNetDecoder, self).__init__(name=name)
        self.setting = setting    
        self.activation = activation  

    def call(self, input_layer):
        num_channels = self.setting.conv_params[-1][-1][-1]
        x = keras.layers.Dense((25*self.setting.num_points),activation=self.activation )(input_layer)
        x = keras.layers.BatchNormalization(name='%s_bn_1' % (self.name))(x)
        x = keras.layers.Reshape((self.setting.num_points,25), input_shape=(num_channels*self.setting.num_points,))(x) 
        #1D and 2D  Conv layers with kernel and stride side of 1 are identical operations, but for 2D first need to expand then to squeeze
        x = tf.squeeze(keras.layers.Conv2D(self.setting.num_features*3, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=True, activation =self.activation, kernel_initializer='glorot_normal',
                                    name='%s_conv_0' % name)(tf.expand_dims(x, axis=2)),axis=2)  
        x = keras.layers.BatchNormalization(name='%s_bn_2' % (self.name))(x)
        x = tf.squeeze(keras.layers.Conv2D(setting.num_features*2, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=True, activation =self.activation, kernel_initializer='glorot_normal',
                                    name='%s_conv_2' % name)(tf.expand_dims(x, axis=2)),axis=2)  
        x = keras.layers.BatchNormalization(name='%s_bn_3' % (self.name))(x)
        out = tf.squeeze(keras.layers.Conv2D(setting.num_features, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=True, activation =self.activation, kernel_initializer='glorot_normal',
                                    name='%s_conv_out' % self.name)(tf.expand_dims(x, axis=2)),axis=2) 
        return out

    def get_config(self):
        config = super(ParticleNetDecoder, self).get_config()
        config.update({'activation': self.activation, 'name': self.name})
        return config      



