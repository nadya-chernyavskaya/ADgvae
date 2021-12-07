"""
    Model definitions.
    Models are largely taken from https://github.com/ucsd-hep-ex/GraphAE/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter
from torch_geometric.nn import MetaLayer, EdgeConv, GATConv, GATv2Conv, global_mean_pool, DynamicEdgeConv, BatchNorm
from torch_geometric.nn import PointTransformerConv
from torch.autograd import Variable
from torch.nn import ModuleList

import DarkFlow.darkflow.networks.flows as flows


# GNN AE using EdgeConv (mean aggregation graph operation). Basic GAE model.
class EdgeNet(nn.Module):
    def __init__(self, input_dim=7, output_dim=4, big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNet, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, hidden_dim),
                               nn.ReLU(),
        )
        
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, output_dim)
        )
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)

    def forward(self, data):
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        x = self.decoder(x,data.edge_index)
        return x



# GVAE based on EdgeNet model above.
class EdgeNetVAE(nn.Module):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2, aggr='mean'):
        super(EdgeNetVAE, self).__init__()
        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(big_dim, hidden_dim)
        self.var_layer = nn.Linear(big_dim, hidden_dim)
        
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, output_dim)
        )
        
        #self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, data):
        #x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        mu = self.mu_layer(x)
        log_var = self.var_layer(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z,data.edge_index)
        return x, mu, log_var



class PlanarVAE(nn.Module):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2, activation=nn.ReLU(),num_flows=6):
        super(PlanarVAE, self).__init__()

        self.hidden_dim = hidden_dim

        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, hidden_dim),
                               nn.ReLU(),
        )
        
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, output_dim)
        )

        self.encoder_1 = EdgeConv(nn=encoder_nn,aggr='mean')
        self.decoder_1 = EdgeConv(nn=decoder_nn,aggr='mean')

        self.mu_layer = nn.Linear(big_dim, hidden_dim)
        self.var_layer = nn.Linear(big_dim, hidden_dim)
        #self.batchnorm = nn.BatchNorm1d(input_dim)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0
        # Flow parameters
        flow = flows.Planar
        self.num_flows = num_flows
        # Amortized flow parameters
        self.amor_u = nn.Linear(big_dim, self.num_flows * self.hidden_dim)
        self.amor_w = nn.Linear(big_dim, self.num_flows * self.hidden_dim)
        self.amor_b = nn.Linear(big_dim, self.num_flows)
        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x, edge_index):
        x = self.encoder_1(x,edge_index)
        return x

    def decode(self, x, edge_index):
        x_decoded = self.decoder_1(x,edge_index)
        return x

    def encode_plus_flows(self, x, edge_index):
        batch_size = x.size(0)
        x = self.encode(x,edge_index)
        mu = self.mu_layer(x)
        log_var = self.var_layer(x)      

        # return amortized u an w for all flows
        u = self.amor_u(x).view(batch_size, self.num_flows, self.hidden_dim, 1)
        w = self.amor_w(x).view(batch_size, self.num_flows, 1, self.hidden_dim)
        b = self.amor_b(x).view(batch_size, self.num_flows, 1, 1)

        return mu, log_var, u, w, b



    def forward(self, data):
        self.log_det_j = 0
        #x = self.batchnorm(data.x)
        x = data.x
        mu, log_var, u, w, b = self.encode_plus_flows(x,data.edge_index)

        # Sample z_0
        z = [self.reparameterize(mu, log_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k)) #planar.'flow_'+k
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_decoded = self.decode(z[-1],data.edge_index)

        return x_decoded, mu, log_var, self.log_det_j, z[0], z[-1]


class PlanarVAE_EdgeNet(PlanarVAE):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2, aggr='mean', activation=nn.ReLU(),num_flows=6):
        super().__init__(input_dim=input_dim, output_dim=output_dim,  big_dim=big_dim, hidden_dim=hidden_dim,activation=activation,num_flows=num_flows)

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim),
                                   activation,
                                   nn.BatchNorm1d(big_dim)
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   activation,
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   activation,
                                   nn.BatchNorm1d(big_dim)
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   activation,
                                   nn.Linear(big_dim, big_dim),
                                   nn.BatchNorm1d(big_dim),
                                   activation,
                                   nn.Linear(big_dim, big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2)
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim*2), big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, output_dim)
        )

        self.encoder_1, self.decoder_1 = None, None
        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)
    

    def encode(self, x, edge_index):
        x = self.encoder_1(x,edge_index)
        x = self.encoder_2(x,edge_index)
        return x

    def decode(self, x, edge_index):
        x_decoded = self.decoder_1(x,edge_index)
        x_decoded = self.decoder_2(x_decoded,edge_index)
        return x_decoded


class TriangularSylvesterVAE(nn.Module):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2,activation=nn.ReLU(),num_flows=6):
        super(TriangularSylvesterVAE, self).__init__()

        self.hidden_dim = hidden_dim


        encoder_nn = nn.Sequential(nn.Linear(2*(input_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU()
        )
        decoder_nn = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, big_dim),
                               nn.ReLU(),
                               nn.Linear(big_dim, output_dim)
        )
        self.encoder_1 = EdgeConv(nn=encoder_nn,aggr='mean')
        self.decoder_1 = EdgeConv(nn=decoder_nn,aggr='mean')

        self.mu_layer = nn.Linear(big_dim, hidden_dim)
        self.var_layer = nn.Linear(big_dim, hidden_dim)
        #self.batchnorm = nn.BatchNorm1d(input_dim)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0
        # Flow parameters
        flow = flows.TriangularSylvester
        self.num_flows = num_flows
        # permuting indices corresponding to Q=P (permutation matrix) for every other flow
        flip_idx = torch.arange(self.hidden_dim - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.hidden_dim, self.hidden_dim), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.hidden_dim).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(big_dim, self.num_flows * self.hidden_dim * self.hidden_dim)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(big_dim, self.num_flows * self.hidden_dim),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(big_dim, self.num_flows * self.hidden_dim),
            self.diag_activation
        )

        self.amor_b = nn.Linear(big_dim, self.num_flows * self.hidden_dim)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.hidden_dim)

            self.add_module('flow_' + str(k), flow_k)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x, edge_index):
        x = self.encoder_1(x,edge_index)
        return x        

    def encode_plus_flows(self, x, edge_index):
        batch_size = x.size(0)
        x = self.encode(x,edge_index)
        mu = self.mu_layer(x)
        log_var = self.var_layer(x)      

        # Amortized r1, r2, b for all flows
        full_d = self.amor_d(x)
        diag1 = self.amor_diag1(x)
        diag2 = self.amor_diag2(x)

        full_d = full_d.view(batch_size, self.hidden_dim, self.hidden_dim, self.num_flows)
        diag1 = diag1.view(batch_size, self.hidden_dim, self.num_flows)
        diag2 = diag2.view(batch_size, self.hidden_dim, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        b = self.amor_b(x)
          # Resize flow parameters to divide over K flows
        b = b.view(batch_size, 1, self.hidden_dim, self.num_flows)

        return mu, log_var, r1, r2, b

    def decode(self, x, edge_index):
        x = self.decoder_1(x,edge_index)
        return x   


    def forward(self, data):
        self.log_det_j = 0
        #x = self.batchnorm(data.x)
        x = data.x
        mu, log_var,  r1, r2, b = self.encode_plus_flows(x,data.edge_index)

        # Sample z_0
        z = [self.reparameterize(mu, log_var)]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            if k % 2 == 1:
                # Alternate with reorderering z for triangular flow
                permute_z = self.flip_idx
            else:
                permute_z = None

            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], b[:, :, :, k], permute_z, sum_ldj=True)

            z.append(z_k)
            self.log_det_j += log_det_jacobian


        x_decoded = self.decode(z[-1],data.edge_index)

        return x_decoded, mu, log_var, self.log_det_j, z[0], z[-1]


class TriangularSylvesterVAE_EdgeNet(TriangularSylvesterVAE):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2, aggr='mean', activation=nn.ReLU(),num_flows=6):
        super().__init__(input_dim=input_dim, output_dim=output_dim,  big_dim=big_dim, hidden_dim=hidden_dim,activation=activation,num_flows=num_flows)

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim),
                                   activation,
                                   nn.BatchNorm1d(big_dim)
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   activation,
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, big_dim),
                                   activation,
                                   nn.BatchNorm1d(big_dim)
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   activation,
                                   nn.Linear(big_dim, big_dim),
                                   nn.BatchNorm1d(big_dim),
                                   activation,
                                   nn.Linear(big_dim, big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2)
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim*2), big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Linear(big_dim*2, output_dim)
        )

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)


    def encode(self, x, edge_index):
        x = self.encoder_1(x,edge_index)
        x = self.encoder_2(x,edge_index)
        return x

    def decode(self, x, edge_index):
        x_decoded = self.decoder_1(x,edge_index)
        x_decoded = self.decoder_2(x_decoded,edge_index)
        return x_decoded




class TriangularSylvesterVAE_GATStack(TriangularSylvesterVAE):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2,num_conv_layers=5,heads=1,dropout=0.,negative_slope=0.2, activation=nn.ReLU(),num_flows=6):
        super().__init__(input_dim=input_dim, output_dim=output_dim,  big_dim=big_dim, hidden_dim=hidden_dim,activation=activation,num_flows=num_flows)

        self.activation = activation
        self.dropout = dropout
        self.enc_convs = ModuleList()
        self.enc_convs.append(
            GATv2Conv(
                in_channels=input_dim,
                out_channels=big_dim,
                heads=heads,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=False
            )
        )
        for _ in range(num_conv_layers - 2):
            conv = GATv2Conv(
                in_channels=big_dim * heads,
                out_channels=big_dim,
                heads=heads,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=False
            )
            self.enc_convs.append(conv)
        conv = GATv2Conv(
            in_channels=big_dim * heads,
            out_channels=big_dim,
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=False,
            concat=False,
        )
        self.enc_convs.append(conv)
        self.encoder_1 = None
        self.decoder_1 = None


        self.dec_convs = ModuleList()
        conv = GATv2Conv(
            in_channels=hidden_dim, 
            out_channels=big_dim, 
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=False
        )
        self.dec_convs.append(conv)
        conv = GATv2Conv(
            in_channels=big_dim*heads, 
            out_channels=big_dim, 
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=False
        )        
        self.dec_convs.append(conv)
        conv = GATv2Conv(
            in_channels=big_dim*heads, 
            out_channels=output_dim, 
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=False,
            concat=False
        )
        self.dec_convs.append(conv)

    def encode(self, x, edge_index):
        for conv in self.enc_convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.enc_convs[-1](x, edge_index)
        return x

    def decode(self, x, edge_index):
        for conv in self.dec_convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dec_convs[-1](x, edge_index) 
        return x


class TriangularSylvesterVAE_EdgeAttention(TriangularSylvesterVAE):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2,num_conv_layers=3,heads=3,dropout=0.1,negative_slope=0.2, activation=nn.ReLU(),num_flows=6):
        super().__init__(input_dim=input_dim, output_dim=output_dim,  big_dim=big_dim, hidden_dim=hidden_dim,activation=activation,num_flows=num_flows)


        self.activation = activation
        self.dropout = dropout
        aggr = 'mean'

        encoder_nn_1 = nn.Sequential(nn.Linear(2*(input_dim), big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(big_dim*2, big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(big_dim*2, big_dim),
                                   activation,
                                   nn.BatchNorm1d(big_dim),
                                   nn.Dropout(p=self.dropout)
        )
        encoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim), big_dim),
                                   activation,
                                   nn.BatchNorm1d(big_dim),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(big_dim, int(big_dim/2)),
                                   activation,
                                   nn.BatchNorm1d(int(big_dim/2)),
        )
        decoder_nn_1 = nn.Sequential(nn.Linear(2*(hidden_dim), big_dim),
                                   activation,
                                   nn.Linear(big_dim, big_dim),
                                   nn.BatchNorm1d(big_dim),
                                   nn.Dropout(p=self.dropout),
                                   activation,
                                   nn.Linear(big_dim, big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Dropout(p=self.dropout)
        )
        decoder_nn_2 = nn.Sequential(nn.Linear(2*(big_dim*2), big_dim*2),
                                   activation,
                                   nn.BatchNorm1d(big_dim*2),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(big_dim*2, big_dim),
                                   activation,
                                   nn.BatchNorm1d(big_dim),
                                   nn.Linear(big_dim, output_dim)
        )

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)


        self.enc_convs = ModuleList()
        self.enc_convs.append(
            GATv2Conv(
                in_channels=input_dim,
                out_channels=big_dim,
                heads=heads,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=True
            )
        )
        for _ in range(num_conv_layers - 2):
            conv = GATv2Conv(
                in_channels=big_dim * heads,
                out_channels=big_dim,
                heads=heads,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=True
            )
            self.enc_convs.append(conv)
        conv = GATv2Conv(
            in_channels=big_dim * heads,
            out_channels=int(big_dim/2),
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=True,
            concat=False,
        )
        self.enc_convs.append(conv)

        self.dec_convs = ModuleList()
        conv = GATv2Conv(
            in_channels=hidden_dim, 
            out_channels=big_dim, 
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=True
        )
        self.dec_convs.append(conv)
        conv = GATv2Conv(
            in_channels=big_dim*heads, 
            out_channels=big_dim, 
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=True
        )        
        self.dec_convs.append(conv)
        conv = GATv2Conv(
            in_channels=big_dim*heads, 
            out_channels=output_dim, 
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=True,
            concat=False
        )
        self.dec_convs.append(conv)

        self.lin_out = nn.Linear(output_dim*2, output_dim) #if concat, then *2
        self.lin_out_2 = nn.Linear(output_dim, output_dim) #if concat, then *2


    def encode(self, x, edge_index):
        x_cloud = self.encoder_1(x,edge_index)
        x_cloud = self.encoder_2(x_cloud,edge_index)

        x_gat = x
        for conv in self.enc_convs[:-1]:
            x_gat = conv(x_gat, edge_index)
            x_gat = self.activation(x_gat)
        x_gat = self.enc_convs[-1](x_gat, edge_index) 

        x_tot = torch.cat((x_cloud, x_gat), dim=1)
        #x_tot = torch.add(x_cloud, x_gat)
        return x_tot


    def decode(self, x, edge_index):
        x_cloud = self.decoder_1(x,edge_index)
        x_cloud = self.decoder_2(x_cloud,edge_index)

        x_gat = x
        for conv in self.dec_convs[:-1]:
            x_gat = conv(x_gat, edge_index)
            x_gat = self.activation(x_gat)
        x_gat = self.dec_convs[-1](x_gat, edge_index) 

        x_tot = torch.cat((x_cloud, x_gat), dim=1)
        x_tot = self.lin_out(x_tot) 
        x_tot = self.activation(x_tot)
        x_tot = self.lin_out_2(x_tot) 

        #x_tot = torch.add(x_cloud, x_gat)
        return x_tot



class GATLayer(nn.Module):   
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm,concat,  activation=F.relu,add_self_loops=True,negative_slope=0.2):
        super().__init__()
        self.activation = activation
        self.batch_norm = batch_norm
            

        self.gatconv = GATv2Conv(
            in_channels=in_dim, 
            out_channels=out_dim, 
            heads=num_heads,
            negative_slope=negative_slope,
            dropout=dropout, #dropout of attention coefficients
            add_self_loops=add_self_loops,
            concat=concat
        )  

        if self.batch_norm:
            if concat :
                self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)
            else : 
                self.batchnorm_h = nn.BatchNorm1d(out_dim)    

    def forward(self, feature, edge_index):
        h = self.gatconv(feature, edge_index)
            
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation:
            h = self.activation(h)

        return h


class EdgeConvLayer(nn.Module):   
    def __init__(self, in_dim, out_dim, dropout, batch_norm, activation=F.relu, aggr='mean'):
        super().__init__()
        self.activation = activation
        self.batch_norm = batch_norm
            
        if self.batch_norm:
            self.edgeconv = nn.Sequential(nn.Linear(2*(in_dim), out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   activation,
                                   nn.Dropout(p=dropout)) 
        else :
            self.edgeconv = nn.Sequential(nn.Linear(2*(in_dim), out_dim),
                                   activation,
                                   nn.Dropout(p=dropout))             

        ###dropout in AE as a regularization 
        self.edgeconv = EdgeConv(nn=self.edgeconv,aggr=aggr)

    def forward(self, feature, edge_index):
        h = self.edgeconv(feature, edge_index)
    
        return h


def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(
            nn.Linear(channels[i - 1], channels[i]),
            nn.BatchNorm1d (channels[i]) if batch_norm else nn.Identity(),
            nn.ReLU()
        )
        for i in range(1, len(channels))
    ])



class PointTransformerLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels,pos_dim=2,batch_norm=True, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.lin_in = nn.Linear(in_channels, in_channels)
        self.lin_out = nn.Linear(out_channels, out_channels)

        self.pos_nn = MLP([pos_dim, out_channels], batch_norm=True)

        self.attn_nn = MLP([out_channels,out_channels], batch_norm=True)

        self.transformer = PointTransformerConv(
            in_channels,
            out_channels,
            pos_nn=self.pos_nn,
            attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x)
        x = self.activation(x)
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x)
        x = self.activation(x)
        return x



class TriangularSylvesterVAE_EdgeAttentionInception(TriangularSylvesterVAE):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2,num_conv_layers=3,heads=3,dropout=0.1,negative_slope=0.2, activation=nn.ReLU(),num_flows=6):
        super().__init__(input_dim=input_dim, output_dim=output_dim,  big_dim=big_dim, hidden_dim=hidden_dim,activation=activation,num_flows=num_flows)


        self.encoder_1, self.decoder_1 = None, None
        self.activation = activation
        self.num_conv_layers = num_conv_layers
        aggr = 'mean'

        '''Edge Conv encoder part '''
        self.enc_edge_convs = ModuleList()
        self.enc_edge_convs.append(
            EdgeConvLayer(in_dim = input_dim, out_dim = big_dim, 
                dropout = dropout, batch_norm = True, activation=activation)
        )
        for _ in range(num_conv_layers - 2):
            conv = EdgeConvLayer(in_dim = big_dim *(1+heads), out_dim = big_dim, 
                dropout = dropout, batch_norm = True, activation=activation)
            self.enc_edge_convs.append(conv)
        self.enc_edge_convs.append(EdgeConvLayer(in_dim = big_dim *(1+heads), out_dim = int(big_dim/2), 
                dropout = dropout, batch_norm = True, activation=activation)
        )
        '''Edge Conv decoder part '''
        self.dec_edge_convs = ModuleList()
        self.dec_edge_convs.append(
            EdgeConvLayer(in_dim = hidden_dim, out_dim = big_dim, 
                dropout = dropout, batch_norm = True, activation=activation)
        )
        for _ in range(num_conv_layers - 2):
            conv = EdgeConvLayer(in_dim = big_dim *(1+heads), out_dim = big_dim, 
                dropout = dropout, batch_norm = True, activation=activation)
            self.dec_edge_convs.append(conv)
        self.dec_edge_convs.append(EdgeConvLayer(in_dim = big_dim *(1+heads), out_dim = output_dim, 
                dropout = dropout, batch_norm = True, activation=activation)
        )

        ''' GAT encoder part '''
        self.enc_gat_convs = ModuleList()
        self.enc_gat_convs.append(
            GATLayer(in_dim = input_dim, out_dim = big_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True, activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = True
            )
        )
        for _ in range(num_conv_layers - 2):
            conv = GATLayer(in_dim = big_dim *(1+heads), out_dim = big_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True,activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = True
            )
            self.enc_gat_convs.append(conv)
        self.enc_gat_convs.append(
            GATLayer(in_dim = big_dim *(1+heads), out_dim = int(big_dim/2), 
                num_heads = heads, dropout = dropout, batch_norm = True,activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = False
            ))

        ''' GAT decoder part '''
        self.dec_gat_convs = ModuleList()
        self.dec_gat_convs.append(GATLayer(in_dim = hidden_dim, out_dim = big_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True,activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = True
            ))
        for _ in range(num_conv_layers - 2):
            conv = GATLayer(in_dim = big_dim *(1+heads), out_dim = big_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True,activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = True
            )
            self.dec_gat_convs.append(conv)       
        self.dec_gat_convs.append(GATLayer(in_dim = big_dim *(1+heads), out_dim = output_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True,activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = False
            ))

        self.lin_out = nn.Linear(output_dim*2, output_dim) #if concat, then *2
        self.lin_out_2 = nn.Linear(output_dim, output_dim) #if concat, then *2


    def encode(self, x, edge_index):
        x_edge = self.enc_edge_convs[0](x,edge_index)
        x_gat = self.enc_gat_convs[0](x,edge_index)
        x_tot = torch.cat((x_edge, x_gat), dim=1)

        for i_layer in range(1,self.num_conv_layers - 1,1):

            gat_conv = self.enc_gat_convs[i_layer]
            x_gat = gat_conv(x_tot, edge_index)

            edge_conv = self.enc_edge_convs[i_layer]
            x_edge = edge_conv(x_tot, edge_index)

            x_tot = torch.cat((x_edge, x_gat), dim=1)

        x_gat = self.enc_gat_convs[-1](x_tot, edge_index) 
        x_edge = self.enc_edge_convs[-1](x_tot, edge_index) 
        x_tot = torch.cat((x_edge, x_gat), dim=1)

        return x_tot


    def decode(self, x, edge_index):
        x_edge = self.dec_edge_convs[0](x,edge_index)
        x_gat = self.dec_gat_convs[0](x,edge_index)
        x_tot = torch.cat((x_edge, x_gat), dim=1)

        for i_layer in range(1,self.num_conv_layers - 1,1):

            gat_conv = self.dec_gat_convs[i_layer]
            x_gat = gat_conv(x_tot, edge_index)

            edge_conv = self.dec_edge_convs[i_layer]
            x_edge = edge_conv(x_tot, edge_index)

            x_tot = torch.cat((x_edge, x_gat), dim=1)

        x_gat = self.dec_gat_convs[-1](x_tot, edge_index) 
        x_edge = self.dec_edge_convs[-1](x_tot, edge_index) 
        x_tot = torch.cat((x_edge, x_gat), dim=1)

        x_tot = self.lin_out(x_tot) 
        x_tot = self.activation(x_tot)
        x_tot = self.lin_out_2(x_tot) 

        return x_tot




class TriangularSylvesterVAE_PointTransformer(TriangularSylvesterVAE):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2,num_conv_layers=3, heads=5,dropout=0.05,negative_slope=0.2, activation=nn.ReLU(),num_flows=6):
        super().__init__(input_dim=input_dim, output_dim=output_dim,  big_dim=big_dim, hidden_dim=hidden_dim,activation=activation,num_flows=num_flows)

        self.encoder_1, self.decoder_1 = None, None
        self.activation = activation
        self.num_conv_layers = num_conv_layers

        '''Point Transformer encoder part '''
        self.enc_point_convs = ModuleList()
        self.enc_point_convs.append(PointTransformerLayer(in_channels=input_dim, out_channels=big_dim,batch_norm = True, activation=activation))
        self.enc_point_convs.append(PointTransformerLayer(in_channels=big_dim, out_channels=big_dim*2,batch_norm = True, activation=activation))
        self.enc_point_convs.append(PointTransformerLayer(in_channels=big_dim*2, out_channels=big_dim*2,batch_norm = True, activation=activation))

        '''GAT encoder part '''
        self.enc_gat_convs = ModuleList()
        self.enc_gat_convs.append(
            GATLayer(in_dim = big_dim*2, out_dim = big_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True, activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = True
            )
        )
        self.enc_gat_convs.append(
            GATLayer(in_dim = big_dim *heads, out_dim = big_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True,activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = False
            ))


        ''' GAT decoder part '''
        self.dec_gat_convs = ModuleList()
        self.dec_gat_convs.append(GATLayer(in_dim = hidden_dim, out_dim = big_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True,activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = True
            ))
        for _ in range(num_conv_layers - 2):
            conv = GATLayer(in_dim = big_dim *heads, out_dim = big_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True,activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = True
            )
            self.dec_gat_convs.append(conv)       
        self.dec_gat_convs.append(GATLayer(in_dim = big_dim *heads, out_dim = output_dim, 
                num_heads = heads, dropout = dropout, batch_norm = True,activation=activation,negative_slope=negative_slope,add_self_loops=True,
                concat = False
            ))


    def encode(self, x, edge_index):
        x_pos = x[:,[5,6]] #get eta, phi, #TO DO : should not be hardcoded..
        x_out = self.enc_point_convs[0](x, x_pos, edge_index)

        for i_layer in range(1,len(self.enc_point_convs)):
            transformer = self.enc_point_convs[i_layer]
            x_out = transformer(x_out, x_pos, edge_index)

        for i_layer in range(len(self.enc_gat_convs)):
            gat_conv = self.enc_gat_convs[i_layer]
            x_out = gat_conv(x_out, edge_index)

        return x_out

    def decode(self, x, edge_index):
        x_gat = self.dec_gat_convs[0](x,edge_index)

        for i_layer in range(1,len(self.dec_gat_convs),1):
            gat_conv = self.dec_gat_convs[i_layer]
            x_gat = gat_conv(x_gat, edge_index)

        return x_gat



