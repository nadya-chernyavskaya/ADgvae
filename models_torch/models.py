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
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MetaLayer, EdgeConv, global_mean_pool, DynamicEdgeConv
from torch.autograd import Variable

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
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder = EdgeConv(nn=encoder_nn,aggr=aggr)
        self.decoder = EdgeConv(nn=decoder_nn,aggr=aggr)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, data):
        x = self.batchnorm(data.x)
        x = self.encoder(x,data.edge_index)
        mu = self.mu_layer(x)
        log_var = self.var_layer(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z,data.edge_index)
        return x, mu, log_var



class PlanarEdgeNetVAE(nn.Module):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2, aggr='mean', activation=nn.ReLU(),num_flows=6):
        super(PlanarEdgeNetVAE, self).__init__()

        self.hidden_dim = hidden_dim

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

        self.mu_layer = nn.Linear(big_dim, hidden_dim)
        self.var_layer = nn.Linear(big_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)
    

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
        batch_size = x.size(0)
        x = self.encoder_1(x,edge_index)
        x = self.encoder_2(x,edge_index)
        mu = self.mu_layer(x)
        log_var = self.var_layer(x)      

        # return amortized u an w for all flows
        u = self.amor_u(x).view(batch_size, self.num_flows, self.hidden_dim, 1)
        w = self.amor_w(x).view(batch_size, self.num_flows, 1, self.hidden_dim)
        b = self.amor_b(x).view(batch_size, self.num_flows, 1, 1)

        return mu, log_var, u, w, b



    def forward(self, data):
        self.log_det_j = 0
        x = self.batchnorm(data.x)
        mu, log_var, u, w, b = self.encode(x,data.edge_index)

        # Sample z_0
        z = [self.reparameterize(mu, log_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k)) #planar.'flow_'+k
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_decoded = self.decoder_1(z[-1],data.edge_index)
        x_decoded = self.decoder_2(x_decoded,data.edge_index)

        return x_decoded, mu, log_var, self.log_det_j, z[0], z[-1]



class TriangularSylvesterEdgeNetVAE(nn.Module):
    def __init__(self, input_dim=4, output_dim=4,  big_dim=32, hidden_dim=2, aggr='mean', activation=nn.ReLU(),num_flows=6):
        super(TriangularSylvesterEdgeNetVAE, self).__init__()

        self.hidden_dim = hidden_dim

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

        self.mu_layer = nn.Linear(big_dim, hidden_dim)
        self.var_layer = nn.Linear(big_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.encoder_1 = EdgeConv(nn=encoder_nn_1,aggr=aggr)
        self.encoder_2 = EdgeConv(nn=encoder_nn_2,aggr=aggr)
        self.decoder_1 = EdgeConv(nn=decoder_nn_1,aggr=aggr)
        self.decoder_2 = EdgeConv(nn=decoder_nn_2,aggr=aggr)
    

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
        batch_size = x.size(0)
        x = self.encoder_1(x,edge_index)
        x = self.encoder_2(x,edge_index)
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



    def forward(self, data):
        self.log_det_j = 0
        x = self.batchnorm(data.x)
        mu, log_var,  r1, r2, b = self.encode(x,data.edge_index)

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

        x_decoded = self.decoder_1(z[-1],data.edge_index)
        x_decoded = self.decoder_2(x_decoded,data.edge_index)

        return x_decoded, mu, log_var, self.log_det_j, z[0], z[-1]






