import tensorflow as tf
import tensorflow.keras.layers as klayers
import models.losses as losses
import models.layers as layers
from keras import backend as K
import models.PNmodel as pn


class GraphAutoencoder(tf.keras.Model):

    def __init__(self, nodes_n, feat_sz, activation=tf.nn.tanh, **kwargs):
        super(GraphAutoencoder, self).__init__(**kwargs)
        self.nodes_n = nodes_n
        self.feat_sz = feat_sz
        self.input_shape_feat = [self.nodes_n, self.feat_sz]
        self.input_shape_adj = [self.nodes_n, self.nodes_n]
        self.activation = activation
        self.loss_fn = tf.nn.weighted_cross_entropy_with_logits
        self.encoder = self.build_encoder()
        self.decoder = layers.InnerProductDecoder(activation=tf.keras.activations.linear) # if activation sigmoid -> return probabilities from logits
    
    def build_encoder(self):
        ''' reduce feat_sz to 2 '''
        inputs_feat = klayers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = klayers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat
        #feat_sz-1 layers needed to reduce to R^1 
        x = layers.GraphConvolution(output_sz=6, activation=self.activation)(x, inputs_adj)
        x = layers.GraphConvolution(output_sz=8, activation=self.activation)(x, inputs_adj)
        x = layers.GraphConvolution(output_sz=4, activation=self.activation)(x, inputs_adj)
        for output_sz in reversed(range(2, self.feat_sz)):
            x = layers.GraphConvolution(output_sz=output_sz, activation=self.activation)(x, inputs_adj)
        # NO activation before latent space: last graph with linear pass through activation
        x = layers.GraphConvolution(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
        encoder = tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=x)
        encoder.summary()
        return encoder    
    

    def call(self, inputs):
        z = self.encoder(inputs)
        adj_pred = self.decoder(z)
        return z, adj_pred

    def train_step(self, data):
        (X, adj_tilde), adj_orig = data
        # pos_weight = zero-adj / one-adj -> no-edge vs edge ratio
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)

        with tf.GradientTape() as tape:
            z, adj_pred = self((X, adj_tilde))  # Forward pass
            # Compute the loss value (binary cross entropy for a_ij in {0,1})
            loss = self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight)


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)

        z, adj_pred = self((X, adj_tilde), training=False)  # Forward pass
        loss = tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight))
        
        return {'loss' : loss}


class GraphVariationalAutoencoder(GraphAutoencoder):
    
    def __init__(self, nodes_n, feat_sz, activation, **kwargs):
        super(GraphVariationalAutoencoder, self).__init__(nodes_n, feat_sz, activation, **kwargs)
        self.loss_fn_latent = losses.kl_loss

    def build_encoder(self):

        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat

        for output_sz in reversed(range(2, self.feat_sz)):
            x = layers.GraphConvolution(output_sz=output_sz, activation=self.activation)(x, inputs_adj)

        ''' make latent space params mu and sigma in last compression to feat_sz = 1 '''
        self.z_mean = layers.GraphConvolution(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
        self.z_log_var = layers.GraphConvolution(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)

        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(self.z_mean)[0], self.nodes_n, 1))  
        self.z = self.z_mean +  epsilon * tf.exp(0.5 * self.z_log_var)
        
        return tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=[self.z, self.z_mean, self.z_log_var])
    
    
    def call(self, inputs):
        z, z_mean, z_log_var = self.encoder(inputs)
        adj_pred = self.decoder(z)
        return z, z_mean, z_log_var, adj_pred
    
    
    def train_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)


        with tf.GradientTape() as tape:
            z, z_mean, z_log_var, adj_pred = self((X, adj_tilde))  # Forward pass
            # Compute the loss value (binary cross entropy for a_ij in {0,1})
            loss_reco = tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight), axis=(1,2)) # TODO: add regularization
            loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var), axis=1)
            loss = loss_reco + loss_latent

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {'loss' : loss_reco+loss_latent, 'loss_reco': loss_reco, 'loss_latent': loss_latent}


    def test_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)

        z, z_mean, z_log_var, adj_pred = self((X, adj_tilde), training=False)  # Forward pass
        # Compute the loss value (binary cross entropy for a_ij in {0,1})
        loss_reco =  tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight))
        loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
        
        return {'loss' : loss_reco+loss_latent, 'loss_reco': loss_reco, 'loss_latent': loss_latent}
    
    

class GCNVariationalAutoEncoder(GraphAutoencoder):
    
    def __init__(self, nodes_n, feat_sz, activation, latent_dim, beta_kl,kl_warmup_time, **kwargs):
        self.loss_fn_latent = losses.kl_loss
        self.latent_dim = latent_dim
        self.kl_warmup_time = kl_warmup_time
        self.beta_kl = beta_kl 
        self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)
        super(GCNVariationalAutoEncoder , self).__init__(nodes_n, feat_sz, activation, **kwargs)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()



    def build_encoder(self):
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat

        x = layers.GraphConvolutionBias(output_sz=6, activation=self.activation)(x, inputs_adj)
        x = layers.GraphConvolutionBias(output_sz=2, activation=self.activation)(x, inputs_adj)
      #  for output_sz in reversed(range(2, self.feat_sz)):
      #      x = layers.GraphConvolutionBias(output_sz=output_sz, activation=self.activation)(x, inputs_adj) #right now size is 2 x nodes_n

        '''create flatten layer'''
        x = klayers.Flatten()(x) #flattened to 2 x nodes_n
        '''create dense layer #1 '''
        x = klayers.Dense(self.nodes_n, activation=self.activation)(x) #'relu'
        ''' create dense layer #2 to make latent space params mu and sigma in last compression to feat_sz = 1 '''
        self.z_mean = klayers.Dense(self.latent_dim, activation=self.activation)(x) #tf.keras.activations.linear 
        self.z_log_var = klayers.Dense(self.latent_dim, activation=self.activation)(x) #tf.keras.activations.linear 
        batch = tf.shape(self.z_mean)[0]
        dim = tf.shape(self.z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        self.z = self.z_mean + tf.exp(0.5 * self.z_log_var) * epsilon

        encoder =  tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=[self.z, self.z_mean, self.z_log_var])
        encoder.summary()
        return encoder
    

    def build_decoder(self):
        inputs_feat = tf.keras.layers.Input(shape=self.latent_dim, dtype=tf.float32, name='decoder_input_latent_space') 
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='decoder_input_adjacency')
        out = inputs_feat

        out = klayers.Dense(self.nodes_n, activation=self.activation)(out)     #'relu'
        out = klayers.Dense(2*self.nodes_n, activation=self.activation)(out)   #'relu'
        ''' reshape to 2 x nodes_n '''
        out = tf.keras.layers.Reshape((self.nodes_n,2), input_shape=(2*self.nodes_n,))(out) 
        ''' reconstruct ''' 
       # for output_sz in range(2+1, self.feat_sz+1): #TO DO: none of this should be hardcoded , to be fixed
       #     out = layers.GraphConvolutionBias(output_sz=output_sz, activation=self.activation)(out, inputs_adj)
        out = layers.GraphConvolutionBias(output_sz=6, activation=self.activation)(out, inputs_adj)
        out = layers.GraphConvolutionBias(output_sz=self.feat_sz, activation=self.activation)(out, inputs_adj)

        decoder =  tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=out)
        decoder.summary()
        return decoder

    
    def call(self, inputs):
        (X, adj_orig) = inputs
        z, z_mean, z_log_var = self.encoder(inputs)
        features_out = self.decoder( (z, adj_orig) )
        return features_out, z, z_mean, z_log_var
   
    
    def train_step(self, data):
        (X, adj_orig) = data

        with tf.GradientTape() as tape:
            features_out, z, z_mean, z_log_var  = self((X, adj_orig))  # Forward pass
            # Compute the loss value ( Chamfer plus KL)
            loss_reco = tf.math.reduce_mean(losses.threeD_loss(X,features_out))
            loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
            loss = loss_reco + self.beta_kl * self.beta_kl_warmup * loss_latent
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {'loss' : loss, 'loss_reco': loss_reco, 'loss_latent': loss_latent, 'beta_kl_warmup':self.beta_kl_warmup}


    def test_step(self, data):
        (X, adj_orig) = data
        features_out, z, z_mean, z_log_var = self((X, adj_orig), training=False)  # Forward pass
        loss_reco = tf.math.reduce_mean(losses.threeD_loss(X,features_out))
        loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
        loss = loss_reco + self.beta_kl * self.beta_kl_warmup * loss_latent
        return {'loss' : loss, 'loss_reco': loss_reco, 'loss_latent': loss_latent}   


    
class EdgeConvAutoEncoder(tf.keras.Model):

    def __init__(self, nodes_n, feat_sz, k_neighbors, activation, latent_dim, **kwargs):
        super(EdgeConvAutoEncoder, self).__init__(**kwargs)
        self.nodes_n = nodes_n
        self.feat_sz = feat_sz
        self.activation = activation
        self.latent_dim = latent_dim
        self.point_channels = 10    
        self.edge_channels  = 10
        self.k_neighbors = k_neighbors
        self.input_shape_points = [self.nodes_n,self.feat_sz]
        self.input_shape_edges = [self.nodes_n,self.k_neighbors*self.feat_sz]
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        in_points = klayers.Input(shape=self.input_shape_points, name="in_points")
        in_edges = klayers.Input(shape=self.input_shape_edges, name="in_edges")    
        
        # Input point features BatchNormalization 
        h = klayers.BatchNormalization(name='BatchNorm_points')(in_points)
        # Conv1D with kernel_size=nfeatures to implement a MLP like aggregation of 
        #   input point features
        h_points = klayers.Conv1D(self.point_channels, kernel_size=1, strides=1,
                           activation=self.activation,
                           use_bias="True",
                           name='Conv1D_points')(h) 


        # Input edges features BatchNormalization 
        h = klayers.BatchNormalization(name='BatchNorm_edges')(in_edges)
        # Conv1D (MLP like aggregation) of input edge features
        h_edges  = klayers.Conv1D(self.edge_channels, kernel_size=1, strides=1,
                           activation=self.activation,
                           use_bias="True",
                           name='Conv1D_edges')(h)

        # Concatenate points+edge features    
        #h = h_points+h_edges                        #particle net uses sum
        h = tf.concat([h_points,h_edges],axis=2) #Andre uses concatenation

        # Flatten to format for MLP input
        h=klayers.Flatten(name='Flatten')(h)
    
        #Latent dimension
        hidden = klayers.Dense(self.latent_dim, name = 'latent',activation=self.activation )(h)
        encoder = tf.keras.Model(inputs=(in_points,in_edges), outputs=hidden,name='EdgeConvEncoder')
        encoder.summary()
        return encoder 

    def build_decoder(self):
        #Decode from latent dimension
        hidden   = klayers.Input(shape=(self.latent_dim, ), name='decoder_input')
        h = klayers.Dense((self.point_channels+self.edge_channels)*self.nodes_n,activation=self.activation )(hidden)
        h = klayers.Reshape((self.nodes_n,self.point_channels+self.edge_channels), input_shape=((self.point_channels+self.edge_channels)*self.nodes_n,))(h) 
        out = klayers.Conv1D(self.feat_sz, kernel_size=1, strides=1,
                          activation=self.activation,
                          use_bias="True",
                          name='Conv1D_out')(h)

        decoder = tf.keras.Model(inputs=hidden, outputs=out,name='EdgeConvDecoder')
        decoder.summary() 
        return decoder
    

    def call(self, inputs):
        features_out = self.decoder(self.encoder(inputs))
        return features_out

    def train_step(self, data):
        (nodes_feats_in, edge_feats_in) , nodes_feats_in = data

        with tf.GradientTape() as tape:
            nodes_feats_out = self((nodes_feats_in, edge_feats_in))  # Forward pass
            # Compute the loss value 
            loss = tf.math.reduce_mean(losses.threeD_loss(nodes_feats_in,nodes_feats_out))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        (nodes_feats_in, edge_feats_in) , nodes_feats_in = data
        
        nodes_feats_out = self((nodes_feats_in, edge_feats_in), training=False)  # Forward pass
        loss = tf.math.reduce_mean(losses.threeD_loss(nodes_feats_in,nodes_feats_out))
        
        return {'loss' : loss}
    
    

class EdgeConvVariationalAutoEncoder(EdgeConvAutoEncoder):
    def __init__(self, nodes_n, feat_sz,k_neighbors, activation, latent_dim, beta_kl,kl_warmup_time, **kwargs):
        self.latent_dim = latent_dim
        self.kl_warmup_time = kl_warmup_time
        self.beta_kl = beta_kl 
        self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)
        super(EdgeConvVariationalAutoEncoder, self).__init__(nodes_n, feat_sz, k_neighbors,activation,latent_dim, **kwargs)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        in_points = klayers.Input(shape=self.input_shape_points, name="in_points")
        in_edges = klayers.Input(shape=self.input_shape_edges, name="in_edges")    
        
        # Conv1D with kernel_size=nfeatures to implement a MLP like aggregation of 
        #   input point features
        h_points = klayers.Conv1D(self.point_channels, kernel_size=1, strides=1,
                           activation=self.activation,
                           use_bias="True",
                           kernel_initializer='glorot_normal',
                           name='Conv1D_points')(in_points) 


        # Conv1D (MLP like aggregation) of input edge features
        h_edges  = klayers.Conv1D(self.edge_channels, kernel_size=1, strides=1,
                           activation=self.activation,
                           use_bias="True",
                           kernel_initializer='glorot_normal',
                           name='Conv1D_edges')(in_edges)

        # Concatenate points+edge features                           
        h = tf.concat([h_points,h_edges],axis=2)

        # Flatten to format for MLP input
        h=klayers.Flatten(name='Flatten')(h)
    
        #Latent dimension and sampling 
        z_mean = klayers.Dense(self.latent_dim, name = 'z_mean',activation='relu' )(h)
        z_log_var = klayers.Dense(self.latent_dim, name='z_log_var', activation='relu' )(h)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        encoder = tf.keras.Model(inputs=(in_points,in_edges), outputs=[z, z_mean, z_log_var],name='EdgeConvEncoderVAE')
        encoder.summary()
        return encoder

    def build_decoder(self):
        # Decoder Input
        in_z   = klayers.Input(shape=(self.latent_dim, ), name='decoder_input')
        h = klayers.Dense((self.point_channels+self.edge_channels)*self.nodes_n,activation=self.activation)(in_z)
        h = klayers.Reshape((self.nodes_n,self.point_channels+self.edge_channels),
                            input_shape=((self.point_channels+self.edge_channels)*self.nodes_n,))(h) 
        out_feats = klayers.Conv1D(self.feat_sz, kernel_size=1, strides=1,
                           activation=self.activation,
                           use_bias="True",
                           kernel_initializer='glorot_normal',
                           name='Conv1D_out')(h)
        # Instantiate decoder
        decoder = tf.keras.Model(inputs=in_z, outputs=out_feats, name='EdgeConvDecoderVAE')
        decoder.summary()
        return decoder
    

    def call(self, inputs):
        z, z_mean, z_log_var =  self.encoder(inputs)
        features_out = self.decoder(z) 
        return features_out, z, z_mean, z_log_var

    
    def train_step(self, data):
        (nodes_feats_in, edge_feats_in) , nodes_feats_in = data

        with tf.GradientTape() as tape:
            features_out, z, z_mean, z_log_var  = self((nodes_feats_in, edge_feats_in))  # Forward pass
            # Compute the loss value ( Chamfer plus KL)
            loss_reco = tf.math.reduce_mean(losses.threeD_loss(nodes_feats_in,features_out))
            loss_latent = tf.math.reduce_mean(losses.kl_loss(z_mean, z_log_var))
            loss = loss_reco + self.beta_kl  * loss_latent *(1 if self.beta_kl_warmup==0 else self.beta_kl_warmup )
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {'loss' : loss, 'loss_reco': loss_reco, 'loss_latent': loss_latent, 'beta_kl_warmup':self.beta_kl_warmup}


    def test_step(self, data):
        (nodes_feats_in, edge_feats_in) , nodes_feats_in = data
        features_out, z, z_mean, z_log_var = self((nodes_feats_in, edge_feats_in), training=False)  # Forward pass
        loss_reco = tf.math.reduce_mean(losses.threeD_loss(nodes_feats_in,features_out))
        loss_latent = tf.math.reduce_mean(losses.kl_loss(z_mean, z_log_var))
        loss = loss_reco + self.beta_kl  * loss_latent *(1 if self.beta_kl_warmup==0 else self.beta_kl_warmup )
        return {'loss' : loss, 'loss_reco': loss_reco, 'loss_latent': loss_latent}   



class PNVAE(tf.keras.Model):

    def __init__(self,input_shapes,mask,setting, **kwargs):
        super(PNVAE, self).__init__(**kwargs)
        self.input_shapes = input_shapes
        self.setting = setting
        self.latent_dim = setting.latent_dim
        self.kl_warmup_time = setting.kl_warmup_time
        self.beta_kl = setting.beta_kl 
        self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)
        self.particlenet = self.build_particlenet()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()


    def build_particlenet(self):
        points = klayers.Input(name='points', shape=self.input_shapes['points'])
        features = tf.klayers.Input(name='features', shape=self.input_shapes['features']) if 'features' in self.input_shapes else None
        mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None

        inputs =[points, features]
        pool_layer = layers.ParticleNetBase(setting=self.setting, activation=self.activation, name='ParticleNet')(points, features, mask)

        return pool_layer

    def build_encoder(self):
        input_pn   = klayers.Input(shape=(self.setting.conv_params[-1][-1][-1], ), name='encoder_input')

        if 'vae'.lower() in self.setting.ae_type :
            z, z_mean_, z_log_var = _sampling(input_pn, setting=self.setting, name=self.name)
            encoder_output = [z, z_mean_, z_log_var]
        else :  
            latent_space = klayers.Dense(self.setting.latent_dim,activation=self.activation )(input_pn)
            encoder_output = [latent_space]

        return encoder_output



    def call(self, inputs):
        points = tf.keras.Input(name='points', shape=self.input_shapes['points'])
        features = tf.keras.Input(name='features', shape=self.input_shapes['features']) if 'features' in self.input_shapes else None
        inputs =[points, features]
        mask = None
        pool_layer = pn._particle_net_base(points, features, mask, self.setting, activation=klayers.LeakyReLU(alpha=0.1), name='ParticleNet')
        encoder = pn._encoder(pool_layer, setting=self.setting,activation=klayers.LeakyReLU(alpha=0.1), name='encoder')
        decoder = pn._decoder(encoder[0],setting=self.setting,activation=klayers.LeakyReLU(alpha=0.1), name='decoder')
        return tf.keras.Model(inputs=inputs, outputs=decoder, name='ParticleNet'+self.setting.ae_type)


    def train_step(self, data):
        (coord_in, feats_in) , feats_in = data

        with tf.GradientTape() as tape:
            feats_out = self((coord_in, feats_in))  # Forward pass
            # Compute the loss value 
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
        
        feats_out = self((coord_in, feats_in), training=False)  # Forward pass
        loss = tf.math.reduce_mean(losses.threeD_loss(feats_in,feats_out))
        
        return {'loss' : loss}
    
    

    
    
    
class KLWarmupCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(KLWarmupCallback, self).__init__()
        self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        if self.model.kl_warmup_time!=0 : 
            #By design the first epoch will have a small fraction of latent loss
            kl_value = ((epoch+1)/self.model.kl_warmup_time) * (epoch < self.model.kl_warmup_time) + 1.0 * (epoch >= self.model.kl_warmup_time)
        else : 
            kl_value=1
        tf.keras.backend.set_value(self.model.beta_kl_warmup, kl_value)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['beta_kl_warmup'] = tf.keras.backend.get_value(self.model.beta_kl_warmup)


