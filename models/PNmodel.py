import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU


# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    with tf.name_scope('dmat'):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        return D


def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    with tf.name_scope('knn'):
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return tf.gather_nd(features, indices)


def edge_conv(points, features, num_points, K, channels, with_bn=True, activation='relu', pooling='average', name='edgeconv'):
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

    with tf.name_scope('edgeconv'):

        # distance
        D = batch_distance_matrix_general(points, points)  # (N, P, P)
        _, indices = tf.nn.top_k(-D, k=K + 1)  # (N, P, K+1)
        indices = indices[:, :, 1:]  # (N, P, K)

        fts = features
        knn_fts = knn(num_points, K, indices, fts)  # (N, P, K, C)
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, K, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

        x = knn_fts
        for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (name, idx))(x)
            if with_bn:
                x = keras.layers.BatchNormalization(name='%s_bn%d' % (name, idx))(x)
            if activation:
                x = keras.layers.Activation(activation, name='%s_act%d' % (name, idx))(x)

        if pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')

        # shortcut of constituents features
        sc = keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % name)(tf.expand_dims(features, axis=2))
        if with_bn:
            sc = keras.layers.BatchNormalization(name='%s_sc_bn' % name)(sc)
        sc = tf.squeeze(sc, axis=2)

        if activation:
       #     return keras.layers.Activation(activation, name='%s_sc_act' % name)(fts)  # (N, P, C') #try with concatenation instead of sum
            return keras.layers.Activation(activation, name='%s_sc_act' % name)(sc + fts)  # (N, P, C') #try with concatenation instead of sum
        else:
            return sc + fts
       #     return fts #this way there is no shortcut


def _particle_net_base(points, features=None, mask=None, setting=None, name='particle_net'):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optinal

    with tf.name_scope(name):
        if features is None:
            features = points

        if mask is not None:
            mask = tf.cast(tf.not_equal(mask, 0), dtype='float32')  # 1 if valid
            coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype='float32'))  # make non-valid positions to 99

        fts = tf.squeeze(keras.layers.BatchNormalization(name='%s_fts_bn' % name)(tf.expand_dims(features, axis=2)), axis=2)
        for layer_idx, layer_param in enumerate(setting.conv_params):
            K, channels = layer_param
            if mask is not None:
                pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
            else : pts=points
            fts = edge_conv(pts, fts, setting.num_points, K, channels, with_bn=True, activation=LeakyReLU(alpha=0.1),
                            pooling=setting.conv_pooling, name='%s_%s%d' % (name, 'EdgeConv', layer_idx))

        if mask is not None:
            fts = tf.multiply(fts, mask)

        pool = tf.reduce_mean(fts, axis=1)  # (N, C)  #pooling over all jet constituents
        return pool


def _ae_base(pool_layer, setting=None, name='ae'):
     num_channels = setting.conv_params[-1][-1][-1]
     print(num_channels)
     latent_space = keras.layers.Dense(setting.latent_dim,activation=LeakyReLU(alpha=0.1) )(pool_layer)
     x = keras.layers.Dense((25*setting.num_points),activation=LeakyReLU(alpha=0.1) )(latent_space)
     x = keras.layers.BatchNormalization(name='%s_bn_1' % (name))(x)
     x = keras.layers.Reshape((setting.num_points,25), input_shape=(num_channels*setting.num_points,))(x) 
     #1D and 2D  Conv layers with kernel and stride side of 1 are identical operations, but for 2D first need to expand then to squeeze
     x = tf.squeeze(keras.layers.Conv2D(setting.num_features*3, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=True, activation =LeakyReLU(alpha=0.1), kernel_initializer='glorot_normal',
                                 name='%s_conv_0' % name)(tf.expand_dims(x, axis=2)),axis=2)  
     x = keras.layers.BatchNormalization(name='%s_bn_2' % (name))(x)
     x = tf.squeeze(keras.layers.Conv2D(setting.num_features*2, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=True, activation =LeakyReLU(alpha=0.1), kernel_initializer='glorot_normal',
                                 name='%s_conv_2' % name)(tf.expand_dims(x, axis=2)),axis=2)  
     x = keras.layers.BatchNormalization(name='%s_bn_3' % (name))(x)
     out = tf.squeeze(keras.layers.Conv2D(setting.num_features, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=True, activation =LeakyReLU(alpha=0.1), kernel_initializer='glorot_normal',
                                 name='%s_conv_out' % name)(tf.expand_dims(x, axis=2)),axis=2)  
     return out


class _DotDict:
    pass

def get_particle_net_lite_ae(input_shapes):
    r"""ParticleNet-Lite model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (7, (32, 32, 32)),
        (7, (64, 64, 64)),
        ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = 'average'
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = None 
    setting.num_points = input_shapes['points'][0] #num of original consituents
    setting.num_features = input_shapes['features'][1] #num of original features
    setting.latent_dim = 7 

    points = keras.Input(name='points', shape=input_shapes['points'])
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = None
    pool_layer = _particle_net_base(points, features, mask, setting, name='ParticleNet')
    outputs = _ae_base(pool_layer, setting, name='AE')

    return keras.Model(inputs=[points, features], outputs=outputs, name='ParticleNetAE')
