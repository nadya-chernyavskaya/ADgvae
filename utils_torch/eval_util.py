import tqdm
import torch
from torch_geometric.data import Batch
import models_torch.models as models

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multi_gpu = torch.cuda.device_count()>1

def invariant_mass(jet1_e, jet1_px, jet1_py, jet1_pz, jet2_e, jet2_px, jet2_py, jet2_pz):
    """
        Calculates the invariant mass between 2 jets. Based on the formula:
        m_12 = sqrt((E_1 + E_2)^2 - (p_x1 + p_x2)^2 - (p_y1 + p_y2)^2 - (p_z1 + p_z2)^2)
        Args:
            jet1_(e, px, py, pz) (torch.float): 4 momentum of first jet of dijet
            jet2_(e, px, py, pz) (torch.float): 4 momentum of second jet of dijet
        Returns:
            torch.float dijet invariant mass.
    """
    return torch.sqrt(torch.square(jet1_e + jet2_e) - torch.square(jet1_px + jet2_px)
                      - torch.square(jet1_py + jet2_py) - torch.square(jet1_pz + jet2_pz))


def process(data_loader, num_events, model_path, model, loss_ftn_obj, latent_dim, features):

    # load corresponding model
    model = getattr(models, model)(input_dim=input_dim, hidden_dim=latent_dim) 
    modpath = glob(osp.join(model_path,'*.best.pth'))[0]
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(modpath, map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(modpath, map_location=torch.device('cpu')))
    model.eval()

    # Store the return values
    jets_proc_data = []
    input_fts = []
    reco_fts = []

    event = 0
    # for each event in the dataset calculate the loss and inv mass for the leading 2 jets
    with torch.no_grad():
        for k, data_batch in tqdm.tqdm(enumerate(data_loader),total=len(data_loader)):
            # select appropriate features based on what model was trained on

            jets_x_1 = data_batch.x_1
            jets_x_2 = data_batch.x_2
            jets_u_1 = data_batch.u_1
            jets_u_2 = data_batch.u_1
            batch = data_batch.batch

            # run inference on all jets
            if loss_ftn_obj.name == 'vae_loss':
                jets_rec, mu, log_var = model(data_batch)
            else:
                jets_rec = model(data_batch)
            
            # calculate invariant mass (data.u format: p[event_idx, n_particles, jet.mass, jet.px, jet.py, jet.pz, jet.e]])
            dijet_mass = invariant_mass(jets0_u[:,6], jets0_u[:,3], jets0_u[:,4], jets0_u[:,5],
                                        jets1_u[:,6], jets1_u[:,3], jets1_u[:,4], jets1_u[:,5])
            njets = len(torch.unique(batch))
            losses = torch.zeros((njets), dtype=torch.float32)
            # calculate loss per each batch (jet)
            for ib in torch.unique(batch):
                if loss_ftn_obj.name == 'vae_loss':
                    losses[ib] = loss_ftn_obj.loss_ftn(jets_rec[batch==ib], jets_x[batch==ib], mu, log_var)
                elif loss_ftn_obj.name == 'emd_loss':
                    losses[ib] = loss_ftn_obj.loss_ftn(jets_rec[batch==ib], jets_x[batch==ib], torch.tensor(0).repeat(jets_rec[batch==ib].shape[0]))
                else:
                    losses[ib] = loss_ftn_obj.loss_ftn(jets_rec[batch==ib], jets_x[batch==ib])

            loss0 = losses[::2]
            loss1 = losses[1::2]
            jets_info = torch.stack([loss0,
                                     loss1,
                                     dijet_mass,              # mass of dijet
                                     jets0_u[:,2],            # mass of jet 1
                                     jets1_u[:,2],            # mass of jet 2
                                     jets1_u[:,-1]],          # if this event was an anomaly
                                    dim=1)
            jets_proc_data.append(jets_info)
            input_fts.append(jets_x[::2])
            input_fts.append(jets_x[1::2])
            reco_fts.append(jets_rec[::2])
            reco_fts.append(jets_rec[1::2])
            event += njets/2
    # return pytorch tensors
    return torch.cat(jets_proc_data), torch.cat(input_fts), torch.cat(reco_fts)
