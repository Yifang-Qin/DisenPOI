import os, torch, random, argparse, logging, pickle
from utils.dataset import MultiSessionsGraph
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import pandas as pd
from LBS import CTR


ARG = argparse.ArgumentParser()
ARG.add_argument('--epoch', type=int, default=20,
                 help='Epoch num.')
ARG.add_argument('--seed', type=int, default=98765,
                 help='Random seed.')
ARG.add_argument('--batch', type=int, default=1024,
                 help='Training batch size.')
ARG.add_argument('--data', type=str, default='nyc',
                 help='Training dataset.')
ARG.add_argument('--gpu', type=int, default=None,
                 help='Denote training device.')
ARG.add_argument('--patience', type=int, default=5,
                 help='Early stopping patience.')
ARG.add_argument('--embed', type=int, default=64,
                 help='Embedding dimension.')
ARG.add_argument('--gcn_num', type=int, default=2,
                 help='Num of GCN.')
ARG.add_argument('--lr', type=float, default=1e-3,
                 help='Learning rate.')
ARG.add_argument('--beta', type=float, default=0.2,
                 help='Hyper beta for contrastive loss.')
ARG.add_argument('--log', type=str, default=None,
                 help='Log file path.')
ARG.add_argument('--delta', type=str, default='1',
                 help='Disntance graph threshold.')

ARG = ARG.parse_args()


def eval_model(model, dataset, arg):
    loader = DataLoader(dataset, arg.batch, shuffle=True)
    model.eval()
    preds, labels, uids, pois = [], [], [], []

    tars, tar_geos, geo_encs, sess_encs = [], [], [], []
    with torch.no_grad():
        for bn, batch in enumerate(loader):
            logits, con_loss, y, tar_embed, tar_geo_embed, geo_enc_p, sess_enc_p = model(batch.to(device))
            logits = torch.sigmoid(logits)\
                .squeeze().clone().detach().cpu().numpy()
            preds.append(logits)
            labels.append(y.squeeze().cpu().numpy())
            uids.append(batch.uid.cpu().numpy())
    
    preds = np.concatenate(preds, 0, dtype=np.float64)
    labels = np.concatenate(labels, 0, dtype=np.float64)
    uids = np.concatenate(uids, 0)
    auc = roc_auc_score(labels, preds)
    logloss = log_loss(labels, preds)
    return logloss, auc

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_test(model, tr_set, va_set, te_set, arg, device):
    opt = torch.optim.Adam(model.parameters(), lr=arg.lr)
    batch_num = len(tr_set) // arg.batch
    loader = DataLoader(tr_set, arg.batch, shuffle=True)
    best_auc, best_epoch, auc_test, best_loss = 0., 0., 0., 0.

    update_count = 0
    total_anneal_steps = 5 * batch_num

    for epoch in range(arg.epoch):
        model.train()
        for bn, batch in enumerate(loader):
            anneal = (min(arg.beta, arg.beta * update_count / total_anneal_steps))\
                if total_anneal_steps > 0 else arg.beta
            update_count += 1
            
            logits, con_loss, y, tar_embed, tar_geo_embed, geo_enc_p, sess_enc_p = model(batch.to(device))
            bce_loss = torch.nn.BCEWithLogitsLoss()(logits, y.float())
            loss = bce_loss + anneal * con_loss.mean(-1)

            opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)
            # print(torch.norm(model.delta), torch.norm(model.delta.grad))
            opt.step()
            if (bn + 1) % 200 == 0:
                logging.info(f'Batch: {bn + 1} / {batch_num}, loss: {loss.item()} = {bce_loss.item()} + {con_loss.mean(-1).item()}')

        logloss, auc = eval_model(model, va_set, arg)
        logging.info('')
        logging.info(f'Epoch: {epoch + 1} / {arg.epoch}, AUC: {auc}, loss: {logloss}')

        if epoch - best_epoch == arg.patience:
            logging.info(f'Stop training after {arg.patience} epochs without valid improvement.')
            break
        if(logloss < best_loss):
            best_auc = auc
            best_loss = logloss
            best_epoch = epoch
            logloss_test, auc_test = eval_model(model, te_set, arg)

        logging.info(f'Best valid Loss: {best_loss}, AUC: {best_auc} at epch {best_epoch}, test AUC: {auc_test}, loss: {logloss_test}\n')

    logging.info(f'Training finished, best epoch {best_epoch}')
    logging.info(f'Valid AUC: {best_auc}, Loss: {best_loss}, Test AUC: {auc_test}, loss: {logloss_test}')


if __name__ == '__main__':
    seed_torch(ARG.seed)

    LOG_FORMAT = "%(asctime)s  %(message)s"
    DATE_FORMAT = "%m/%d %H:%M"
    if ARG.log is not None:
        logging.basicConfig(filename=ARG.log, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    with open(f'../processed_data/{ARG.data}/raw/val.pkl', 'rb') as f:
        tmp = pickle.load(f)
        n_user, n_poi = pickle.load(f)
        del tmp
    
    dist_threshold = ARG.delta

    train_set = MultiSessionsGraph(f'../processed_data/{ARG.data}', phrase='train')
    val_set = MultiSessionsGraph(f'../processed_data/{ARG.data}', phrase='test')
    test_set = MultiSessionsGraph(f'../processed_data/{ARG.data}', phrase='val')

    with open(f'../processed_data/{ARG.data}/processed/dist_graph_{dist_threshold}.pkl', 'rb') as f:
        dist_edges = torch.LongTensor(pickle.load(f))
        dist_nei = pickle.load(f)

    dist_vec = np.load(f'../processed_data/{ARG.data}/dist_on_graph_{dist_threshold}.npy')

    logging.info(f'Data loaded.')
    logging.info(f'user: {n_user}\tpoi: {n_poi}')
    device = torch.device('cpu') if ARG.gpu is None else torch.device(f'cuda:{ARG.gpu}')
    model = CTR(n_user, n_poi, dist_edges, dist_nei, ARG.embed, ARG.gcn_num, dist_vec, device).to(device)
    train_test(model, train_set, test_set, val_set, ARG, device)

