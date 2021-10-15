import tqdm
import torch
from torch_geometric.data import Batch
import models_torch.models as models

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multi_gpu = torch.cuda.device_count()>1


def train(model, optimizer, loader, total, batch_size, loss_ftn_obj):
    model.train()

    sum_loss = 0.
    sum_loss_reco = 0.
    sum_loss_kl = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        optimizer.zero_grad()

        batch_loss, batch_output = forward_loss(model, data, loss_ftn_obj, device, multi_gpu=False)
        if isinstance(batch_loss,tuple) :
            batch_loss[0].backward() #only total loss from vae
            optimizer.step()
            batch_loss,batch_loss_reco,batch_loss_kl = batch_loss[0].item(),batch_loss[1].item(),batch_loss[2].item()
            sum_loss += batch_loss
            sum_loss_reco += batch_loss_reco
            sum_loss_kl += batch_loss_kl
            t.set_description('train loss tot = %.7f,train loss reco = %.7f,train loss kl = %.7f' %(batch_loss,batch_loss_reco,batch_loss_kl))
            t.refresh() # to show immediately the update
        else:
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.item()
            sum_loss += batch_loss
            t.set_description('train loss = %.7f' % batch_loss)
            t.refresh() # to show immediately the update

    return sum_loss / (i+1), sum_loss_reco / (i+1), sum_loss_kl / (i+1)


# helper to perform correct loss
def forward_loss(model, data, loss_ftn_obj, device, multi_gpu):
    
    if not multi_gpu:
        data = data.to(device)

    if 'emd_loss' in loss_ftn_obj.name or loss_ftn_obj.name == 'chamfer_loss' or loss_ftn_obj.name == 'hungarian_loss':
        batch_output = model(data)
        if multi_gpu:
            data = Batch.from_data_list(data).to(device)
        y = data.x
        batch = data.batch
        batch_loss = loss_ftn_obj.loss_ftn(batch_output, y, batch)

    elif loss_ftn_obj.name == 'emd_in_forward':
        _, batch_loss = model(data)
        batch_loss = batch_loss.mean()

    elif 'vae_loss' in loss_ftn_obj.name :
        batch_output, mu, log_var = model(data)
        y = torch.cat([d.x for d in data]).to(device) if multi_gpu else data.x
        y = y.contiguous()
        batch_loss_tot,batch_loss_reco,batch_loss_kl = loss_ftn_obj.loss_ftn(batch_output, y, mu, log_var)
        batch_loss = (batch_loss_tot,batch_loss_reco,batch_loss_kl)
    else:
        batch_output = model(data)
        y = torch.cat([d.x for d in data]).to(device) if multi_gpu else data.x
        y = y.contiguous()
        batch_loss = loss_ftn_obj.loss_ftn(batch_output, y)

    return batch_loss, batch_output