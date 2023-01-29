import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GatedGraphConv
from torch_geometric.utils import add_self_loops, to_undirected, degree, get_laplacian


class SeqQuery(nn.Module):
    def __init__(self, hidden_size):
        super(SeqQuery, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.alpha = nn.Linear(self.hidden_size, 1)

        for w in self.modules():
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    def forward(self, sess_embed, query, sections):
        v_i = torch.split(sess_embed, sections)
        q_n_repeat = tuple(query[i].view(1, -1).repeat(nodes.shape[0], 1) for i, nodes in enumerate(v_i))

        weight = self.alpha(torch.sigmoid(self.W_1(torch.cat(q_n_repeat, dim=0)) + self.W_2(sess_embed)))
        s_g_whole = weight * sess_embed
        s_g_split = torch.split(s_g_whole, sections)
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        s_h = torch.cat(s_g, dim=0)
        return s_h


class Proj_head(nn.Module):
    def __init__(self, hid_dim):
        super(Proj_head, self).__init__()
        self.hid_dim = hid_dim
        self.lin_head = nn.Linear(hid_dim, hid_dim)
        self.BN = nn.BatchNorm1d(hid_dim)

    def forward(self, x):
        x = self.lin_head(x)
        return self.BN(x)


class CTR(nn.Module):
    def __init__(self, n_user, n_poi, dist_edges, dist_nei, embed_dim, gcn_num, dist_vec, device):
        super(CTR, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.embed_dim = embed_dim
        self.gcn_num = gcn_num
        self.dist_edges = to_undirected(dist_edges, num_nodes=n_poi).to(device)
        self.dist_nei = dist_nei
        self.device = device

        self.poi_embed = nn.Parameter(torch.empty(n_poi, embed_dim))
        self.user_embed = nn.Parameter(torch.empty(n_user, embed_dim))
        nn.init.xavier_normal_(self.poi_embed)
        nn.init.xavier_normal_(self.user_embed)

        self.edge_index, self.adj_weight = get_laplacian(self.dist_edges, normalization='sym', num_nodes=n_poi)
        self.adj_weight = self.adj_weight.to(device)
        self.edge_index = self.edge_index.to(device)

        dist_vec /= dist_vec.max()
        self.dist_vec = torch.cat([torch.Tensor(dist_vec), torch.Tensor(dist_vec), torch.zeros((n_poi,))]).to(device)

        self.sess_conv = GatedGraphConv(self.embed_dim, num_layers=1)
        self.geo_conv = []
        for _ in range(self.gcn_num):
            # self.geo_conv.append(GCN_layer(self.embed_dim, self.embed_dim, device))
            self.geo_conv.append(Geo_GCN(self.embed_dim, self.embed_dim, device))

        self.sess_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.geo_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.CL_builder = Contrastive_BPR()
        self.MLP = nn.Sequential(
                nn.Linear(4 * self.embed_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, 1)
            )

        for w in self.modules():
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

        self.sess_embed = SeqQuery(self.embed_dim)
        self.geo_embed = SeqQuery(self.embed_dim)

    def split_mean(self, section_feat, sections):
        section_embed = torch.split(section_feat, sections)
        mean_embeds = [torch.mean(embeddings, dim=0) for embeddings in section_embed]
        return torch.stack(mean_embeds)

    def forward(self, data):
        sess_idx, edges, batch_idx, uid, tar_poi = data.x.squeeze(), data.edge_index, data.batch, data.uid, data.poi
        sections = tuple(torch.bincount(batch_idx).cpu().numpy())

        # Generate geometric encoding & pooling
        geo_feat = self.poi_embed
        dist_weight = torch.exp(-(self.dist_vec ** 2))
        for i in range(self.gcn_num):
            geo_feat = self.geo_conv[i](geo_feat, self.edge_index, self.adj_weight, dist_weight)
            geo_feat = F.leaky_relu(geo_feat)
            geo_feat = F.normalize(geo_feat, dim=-1)

        geo_enc = self.geo_embed(geo_feat[sess_idx], geo_feat[tar_poi], sections)

        geo_enc_p = self.geo_proj(geo_enc)
        geo_pool = self.geo_proj(self.split_mean(
                    Neigh_pooling()(self.poi_embed, self.edge_index)[sess_idx]
                , sections))

        # Generate session encoding & pooling
        sess_hidden = F.leaky_relu(self.sess_conv(self.poi_embed[sess_idx], edges))
        sess_enc = self.sess_embed(sess_hidden, self.poi_embed[tar_poi], sections)

        sess_enc_p = self.sess_proj(sess_enc)
        sess_pool = self.sess_proj(self.split_mean(self.poi_embed[sess_idx], sections))

        # CL loss for disentanglement
        con_loss = self.CL_builder(geo_enc, geo_pool, sess_pool) + \
                   self.CL_builder(sess_enc, sess_pool, geo_pool)

        tar_embed = self.poi_embed[tar_poi]
        tar_geo_embed = geo_feat[tar_poi]

        logits = self.MLP(torch.cat((tar_embed, tar_geo_embed, sess_enc_p, geo_enc_p), dim=-1)).squeeze()
        return logits, con_loss, data.y, tar_embed, tar_geo_embed, geo_enc_p, sess_enc_p


class Neigh_pooling(MessagePassing):
    def __init__(self, aggr='mean'):
        super(Neigh_pooling, self).__init__(aggr=aggr)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

class Geo_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(Geo_GCN, self).__init__()
        self.W0 = nn.Linear(in_channels, out_channels, bias=False).to(device)
        self.W1 = nn.Linear(in_channels, out_channels, bias=False).to(device)
        self.W2 = nn.Linear(in_channels, out_channels, bias=False).to(device)

        for w in self.modules():
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    def forward(self, x, edge_index, adj_weight, dist_vec):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        dist_weight = torch.exp(-(dist_vec ** 2))

        dist_adj = torch.sparse_coo_tensor(edge_index, dist_weight * norm)
        side_embed = torch.sparse.mm(dist_adj, x)

        bi_embed = torch.mul(x, side_embed)
        return self.W0(side_embed) + self.W1(bi_embed)

class GCN_layer(MessagePassing):
    def __init__(self, in_channels, out_channels, device):
        super(GCN_layer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels).to(device)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index, dist_weight=None):
        edge_index, _ = add_self_loops(edge_index)
        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, embed_dim]
        return norm.view(-1, 1) * x_j


class Contrastive_BPR(nn.Module):
    def __init__(self, beta=1):
        super(Contrastive_BPR, self).__init__()
        self.Activation = nn.Softplus(beta=beta)

    def forward(self, x, pos, neg):
        loss_logit = (x * neg).sum(-1) - (x * pos).sum(-1)
        return self.Activation(loss_logit)
        

def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)
    
    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len, )
    
    return (torch.arange(0, max_len, device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size,max_len)
    .lt(lengths.unsqueeze(1))).reshape(lengths_shape)
