import os,sys
import os.path as osp
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import models_torch.losses as losses
import utils_torch.scaler
from utils_torch.scaler import BasicStandardizer,Standardizer,BasicAndLogStandardizer
sys.path.append(os.path.abspath(os.path.join('..')))
import vande.util.util_plotting as vande_plot


plt_style = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/utils/adfigstyle.mplstyle'
plt.style.use(plt_style)

def loss_distr(losses, save_name):
    """
        Plot distribution of losses
    """
    plt.figure(figsize=(6,4.4))
    plt.hist(losses,bins=np.linspace(0, 600, 101))
    plt.xlabel('Loss', fontsize=16)
    plt.ylabel('Jets', fontsize=16)
    plt.savefig(osp.join(save_name+'.pdf'))
    plt.close()


def plot_reco_difference(input_fts, reco_fts, model_fname, save_path, feature='hadronic'):
    """
    Plot the difference between the autoencoder's reconstruction and the original input
    Args:
        input_fts (numpy array): the original features of the particles
        reco_fts (numpy array): the reconstructed features
    """
    
    if isinstance(input_fts, torch.Tensor):
        input_fts = input_fts.numpy()
    if isinstance(reco_fts, torch.Tensor):
        reco_fts = reco_fts.numpy()

        
    Path(save_path).mkdir(parents=True, exist_ok=True)
  #  label = ['$p_x~[GeV]$', '$p_y~[GeV]$', '$p_z~[GeV]$']
   # feat = ['px', 'py', 'pz']
    label = ['$p_x~[GeV]$', '$p_y~[GeV]$', '$p_z~[GeV]$', '$E~[GeV]$','$p_T$', '$eta$', '$phi$']
    feat = ['px', 'py', 'pz','E','pt', 'eta', 'phi']

    if feature == 'hadronic':# or 'standardized':
        label = ['$p_T$', '$eta$', '$phi$']
        feat = ['pt', 'eta', 'phi']
        
    if feature == 'cartesian':# or 'standardized':
        label = ['$p_x~[GeV]$', '$p_y~[GeV]$', '$p_z~[GeV]$', '$E~[GeV]$']
        feat = ['px', 'py', 'pz','E']
        
    # make a separate plot for each feature
    for i in range(input_fts.shape[1]):
        plt.figure(figsize=(10,8))
        if feature == 'cartesian':
            bins = np.linspace(-20, 20, 101)
            if i == 3:  # different bin size for E momentum
                bins = np.linspace(-5, 35, 101)
        elif feature == 'hadronic':
            bins = np.linspace(-2, 2, 101)
            if i == 0:  # different bin size for pt rel
                bins = np.linspace(-0.05, 0.1, 101)
        elif 'norm' in feature :
            bins = np.linspace(-1, 1, 50)
        elif 'all' in feature :
            bins = np.linspace(-20, 20, 50)
            if i > 3:  # different bin size for hadronic coord
                bins = np.linspace(-2, 2, 50)
            if i == 3:  # different bin size for E momentum
                bins = np.linspace(-5, 35, 50)
            if i == 4:  # different bin size for pt rel
                bins = np.linspace(-2, 10, 50)
        else:
            bins = np.linspace(-1, 1, 50)
        plt.ticklabel_format(useMathText=True)
        plt.hist(input_fts[:,i], bins=bins, alpha=0.5, label='Input', histtype='step', lw=5)
        plt.hist(reco_fts[:,i], bins=bins, alpha=0.5, label='Output', histtype='step', lw=5)
        plt.legend(title='QCD dataset',fontsize=20,bbox_to_anchor=(1., 1.))# fontsize='x-large'
        plt.xlabel(label[i], fontsize='x-large')
        plt.ylabel('Particles', fontsize='x-large')
        plt.tight_layout()
        plt.savefig(osp.join(save_path, feat[i] + '.png'))
        plt.close()
    

@torch.no_grad()
def gen_latent(model, loader, device):
    model.eval()
    mu_fts,log_var_fts = [], []
    z_0_fts, z_last_fts = [],[]
    for t in loader:
        t.to(device)
        out = model(t)#tuple
        if len(out)==6:
            _, mu, log_var, _, z_0, z_last = out
            z_0_fts.append(z_0.cpu().detach())
            z_last_fts.append(z_last.cpu().detach())
        elif len(out)==3:
            _, mu, log_var = out
        mu_fts.append(mu.cpu().detach())
        log_var_fts.append(log_var.cpu().detach())

    mu_fts = torch.cat(mu_fts)
    log_var_fts = torch.cat(log_var_fts)
    if len(out)==6:
        z_0_fts = torch.cat(z_0_fts)
        z_last_fts = torch.cat(z_last_fts)
    return (mu_fts,log_var_fts,z_0_fts,z_last_fts)


@torch.no_grad()
def gen_in_out(model, loader, device):
    model.eval()
    input_fts = []
    reco_fts = []

    for t in loader:
        if isinstance(t, list):
            for d in t:
                input_fts.append(d.x)
        else:
            input_fts.append(t.x)
            t.to(device)

        reco_out = model(t)
        if isinstance(reco_out, tuple):
            reco_out = reco_out[0]
        reco_fts.append(reco_out.cpu().detach())

    input_fts = torch.cat(input_fts)
    reco_fts = torch.cat(reco_fts)
    return input_fts, reco_fts

@torch.no_grad()
def gen_in_out_latent(model, loader, device):
    model.eval()
    mu_fts,log_var_fts = [], []
    z_0_fts, z_last_fts = [],[]
    input_fts = []
    reco_fts = []

    for t in loader:
        t.to(device)
        input_fts.append(t.x)
        out = model(t)#tuple
        if len(out)==6:
            reco, mu, log_var, _, z_0, z_last = out
            z_0_fts.append(z_0.cpu().detach())
            z_last_fts.append(z_last.cpu().detach())
        elif len(out)==3:
            reco, mu, log_var = out
        mu_fts.append(mu.cpu().detach())
        log_var_fts.append(log_var.cpu().detach())
        reco_fts.append(reco.cpu().detach())

    input_fts = torch.cat(input_fts)
    reco_fts = torch.cat(reco_fts)
    mu_fts = torch.cat(mu_fts)
    log_var_fts = torch.cat(log_var_fts)
    if len(out)==6:
        z_0_fts = torch.cat(z_0_fts)
        z_last_fts = torch.cat(z_last_fts)
    return (input_fts,reco_fts,mu_fts,log_var_fts,z_0_fts,z_last_fts)


@torch.no_grad()
def eval_loss(model, loader, device):
    #not finished yet
    model.eval()
    input_fts = []
    reco_fts = []
    reco_fts = []

    for t in loader:
        if isinstance(t, list):
            for d in t:
                input_fts.append(d.x)
        else:
            input_fts.append(t.x)
            t.to(device)
        out = model(t)
        if isinstance(reco_out, tuple):
            reco_out_fts,mu,log_var = reco_out[0],reco_out[1],reco_out[2] #always will be present as first 3

        else : 
            reco_out_fts = out
        reco_fts.append(reco_out_fts.cpu().detach())

    input_fts = torch.cat(input_fts)
    reco_fts = torch.cat(reco_fts)

    return input_fts, reco_fts


def plot_reco_for_loader(model, loader, device, scaler, inverse_scale, model_fname, save_dir, feature_format):
    input_fts, reco_fts = gen_in_out(model, loader, device)
    if 'mseconv' in feature_format:
        reco_fts = losses.xyze_to_ptetaphi_torch(reco_fts)
    save_dir_norm = os.path.join(save_dir, 'normalized/')
    Path(save_dir_norm).mkdir(parents=True, exist_ok=True)
    plot_reco_difference(input_fts, reco_fts, model_fname, save_dir_norm, feature_format)
    if inverse_scale:
        if isinstance(scaler,Standardizer) :
            input_fts = scaler.inverse_transform(input_fts)
            reco_fts = scaler.inverse_transform(reco_fts)
        elif isinstance(scaler,BasicStandardizer) :
            input_fts = scaler.inverse_transform(input_fts)
            reco_fts = scaler.inverse_transform(reco_fts)
        plot_reco_difference(input_fts, reco_fts, model_fname, save_dir, feature_format)



def plot_reco_latent_for_loader(model, loader, device, scaler, inverse_scale, model_fname, save_dir, feature_format):
    input_fts,reco_fts,mu_fts,log_var_fts,z_0_fts,z_last_fts = gen_in_out_latent(model, loader, device)
    vande_plot.plot_2dhist( z_0_fts.numpy(), 'Dim. 0', 'Dim. 1', 'Before Normalizing Flows', plotname=osp.join(save_dir, 'gauss_2d.png'),cmap=plt.cm.Reds)
    vande_plot.plot_2dhist( z_last_fts.numpy() , 'Dim. 0', 'Dim. 1', 'After Normalizing Flows', plotname=osp.join(save_dir, 'normflow_2d.png'),cmap=plt.cm.Reds)
    return 0
    if 'mseconv' in feature_format:
        reco_fts = losses.xyze_to_ptetaphi_torch(reco_fts)
    save_dir_norm = os.path.join(save_dir, 'normalized/')
    Path(save_dir_norm).mkdir(parents=True, exist_ok=True)
    plot_reco_difference(input_fts, reco_fts, model_fname, save_dir_norm, feature_format)
    if inverse_scale:
        if isinstance(scaler,Standardizer) :
            input_fts = scaler.inverse_transform(input_fts)
            reco_fts = scaler.inverse_transform(reco_fts)
        elif isinstance(scaler,BasicStandardizer) or isinstance(scaler,BasicAndLogStandardizer) :
            input_fts = scaler.inverse_transform(input_fts)
            reco_fts = scaler.inverse_transform(reco_fts)
        plot_reco_difference(input_fts, reco_fts, model_fname, save_dir, feature_format)



def loss_curves(epochs, early_stop_epoch, train_loss, valid_loss, save_path, fig_name=''):
    '''
        Graph our training and validation losses.
    '''
    plt.plot(epochs, train_loss, valid_loss)
    plt.xticks(epochs)
    ax = plt.gca()
    ax.set_yscale('log')
    if max(epochs) < 60:
        ax.locator_params(nbins=10, axis='x')
    else:
        ax.set_xticks(np.arange(0, max(epochs), 20))
    if early_stop_epoch != None:
        plt.axvline(x=early_stop_epoch, linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss {}".format(fig_name))
    plt.legend(['Train', 'Validation', 'Best model'])
    plt.savefig(osp.join(save_path, 'loss_curves_{}.pdf'.format(fig_name)))
    plt.savefig(osp.join(save_path, 'loss_curves_{}.png'.format(fig_name)))
    plt.close()


def reco_relative_diff(jet_in, jet_out, save_dir, save_name):
    """
    Plot relative difference between input and predicted features.
    Assumes standard distribution

    :param jet_in: np array [pt, y, phi]
    :param jet_out: np array [pt, y, phi]
    """
    if isinstance(jet_in, torch.Tensor):
        jet_in = jet_in.numpy()
    if isinstance(jet_out, torch.Tensor):
        jet_out = jet_out.numpy()

    rel_diff = (jet_out - jet_in) / (jet_in + 1e-12)

    bins = np.linspace(-1,1, 30)

    plt.hist(rel_diff[:,0], bins=bins)
    feat = 'p_T'
    plt.title(feat)
    plt.savefig(osp.join(save_dir, save_name + '_' + feat))
    plt.close()

    plt.hist(rel_diff[:,1], bins=bins)
    feat = 'eta'
    plt.title(feat)
    plt.savefig(osp.join(save_dir, save_name + '_' + feat))
    plt.close()

    plt.hist(rel_diff[:,2], bins=bins)
    feat = 'phi'
    plt.title(feat)
    plt.savefig(osp.join(save_dir, save_name + '_' + feat))
    plt.close()

def plot_emd_corr(true_emd, pred_emd, save_dir, save_name):
    """
    :param true_emd: np array
    :param pred_emd: np array
    """
    max_range = max(np.max(true_emd), np.max(pred_emd))
    fig, ax = plt.subplots(figsize =(5, 5))
    plt.hist2d(true_emd, pred_emd)
    x_bins = np.linspace(0, max_range, 101)
    y_bins = np.linspace(0, max_range, 101)
    ax.set_xlabel('True EMD')  
    ax.set_ylabel('Pred. EMD')
    plt.savefig(osp.join(save_dir, save_name))
    plt.close()