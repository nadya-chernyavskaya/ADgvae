import tensorflow as tf
import tensorflow.keras.layers as klayers

### Latent Space Loss (KL-Divergence)
@tf.function
def kl_loss(z_mean, z_log_var):
    kl = 1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    return -0.5 * tf.reduce_mean(kl, axis=-1)

### 3D LOSS
@tf.function
def threeD_loss(inputs, outputs): #[batch_size x 100 x 3] -> [batch_size]
    expand_inputs = tf.expand_dims(inputs, 2) # add broadcasting dim [batch_size x 100 x 1 x 3]
    expand_outputs = tf.expand_dims(outputs, 1) # add broadcasting dim [batch_size x 1 x 100 x 3]
    # => broadcasting [batch_size x 100 x 100 x 3] => reduce over last dimension (eta,phi,pt) => [batch_size x 100 x 100] where 100x100 is distance matrix D[i,j] for i all inputs and j all outputs
    distances = tf.math.reduce_sum(tf.math.squared_difference(expand_inputs, expand_outputs), -1)
    # get min for inputs (min of rows -> [batch_size x 100]) and min for outputs (min of columns)
    min_dist_to_inputs = tf.math.reduce_min(distances,1)
    min_dist_to_outputs = tf.math.reduce_min(distances,2)
    return tf.math.reduce_mean(min_dist_to_inputs, 1) + tf.math.reduce_mean(min_dist_to_outputs, 1)

''' GC layers adapted from Kipf: https://github.com/tkipf/gae/blob/0ebbe9b9a8f496eb12deb9aa6a62e7016b5a5ac3/gae/layers.py '''

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
        self.decoder = InnerProductDecoder(activation=tf.keras.activations.linear) # if activation sigmoid -> return probabilities from logits
    
    def build_encoder(self):
        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat
        #feat_sz-1 layers needed to reduce to R^1 
        x = GraphConvolution(output_sz=6, activation=self.activation)(x, inputs_adj)
        x = GraphConvolution(output_sz=8, activation=self.activation)(x, inputs_adj)
        x = GraphConvolution(output_sz=4, activation=self.activation)(x, inputs_adj)
        for output_sz in reversed(range(2, self.feat_sz)):
            x = GraphConvolution(output_sz=output_sz, activation=self.activation)(x, inputs_adj)
        # NO activation before latent space: last graph with linear pass through activation
        x = GraphConvolution(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
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
        self.loss_fn_latent = kl_loss

    def build_encoder(self):

        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat

        for output_sz in reversed(range(2, self.feat_sz)):
            x = GraphConvolution(output_sz=output_sz, activation=self.activation)(x, inputs_adj)

        ''' make latent space params mu and sigma in last compression to feat_sz = 1 '''
        self.z_mean = GraphConvolution(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
        self.z_log_var = GraphConvolution(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)

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

        z, z_mean, z_log_var, adj_pred = self((X, adj_tilde))  # Forward pass
        # Compute the loss value (binary cross entropy for a_ij in {0,1})
        loss_reco =  tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight))
        loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
        
        return {'loss' : loss_reco+loss_latent, 'loss_reco': loss_reco, 'loss_latent': loss_latent}
    
    

class GCNVariationalAutoEncoder(GraphAutoencoder):
    
    def __init__(self, nodes_n, feat_sz, activation, **kwargs):
        super(GCNVariationalAutoEncoder , self).__init__(nodes_n, feat_sz, activation, **kwargs)
        self.loss_fn_latent = kl_loss
        self.latent_dim = 10 #30
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()


    def build_encoder(self):
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat

        for output_sz in reversed(range(2, self.feat_sz)):
            x = GraphConvolution(output_sz=output_sz, activation=self.activation)(x, inputs_adj) #right now size is 2xn_nodes 

        '''create flatten layer'''
        x = klayers.Flatten()(x) #flattened to 2 x n_nodes = 200 
        '''create dense layer #1 '''
        x = klayers.Dense(19, activation='relu')(x) #100
        ''' create dense layer #2 to make latent space params mu and sigma in last compression to feat_sz = 1 '''
        self.z_mean = klayers.Dense(10, activation=tf.keras.activations.linear)(x) #self.latent_dim , 10 
        self.z_log_var = klayers.Dense(10, activation=tf.keras.activations.linear)(x) 
        batch = tf.shape(self.z_mean)[0]
        dim = tf.shape(self.z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        self.z = self.z_mean + tf.exp(0.5 * self.z_log_var) * epsilon

        encoder =  tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=[self.z, self.z_mean, self.z_log_var])
        encoder.summary()
        return encoder
    

    def build_decoder(self):

        inputs_feat = tf.keras.layers.Input(shape=10, dtype=tf.float32, name='decoder_input_latent_space') #latent dim
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='decoder_input_adjacency')
        out = inputs_feat

        out = klayers.Dense(19, activation='relu')(out)   #100 
        out = klayers.Dense(38, activation='relu')(out) #200
        ''' reshape to 2 x n_nodes '''
        out = tf.keras.layers.Reshape((19,2), input_shape=(38,))(out) #200 #self.n_nodes, 19
        ''' reconstruct ''' 
        for output_sz in range(2+1, self.feat_sz+1): #none of this should be hardcoded , to be fixed
            out = GraphConvolution(output_sz=output_sz, activation=self.activation)(out, inputs_adj)

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
            loss_reco = threeD_loss(X,features_out)
            #loss_latent = self.loss_fn_latent(z_mean, z_log_var,self.nodes_n)
            loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
            loss = loss_reco + loss_latent

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {'loss' : loss_reco+loss_latent, 'loss_reco': loss_reco, 'loss_latent': loss_latent}


    def test_step(self, data):
        (X, adj_orig) = data
        features_out, z, z_mean, z_log_var = self((X, adj_orig))  # Forward pass
        loss_reco = threeD_loss(X,features_out)
        #loss_latent = self.loss_fn_latent(z_mean, z_log_var,self.nodes_n)
        loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
        
        return {'loss' : loss_reco+loss_latent, 'loss_reco': loss_reco, 'loss_latent': loss_latent}    
    
