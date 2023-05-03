import numpy as np
import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 全局bias
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, fields):
        u_id = fields[:,0]
        i_id = fields[:,1]
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return torch.sigmoid((U * I).sum(1) + b_u + b_i + self.mean)


class MF_Polar_Align(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=100):
        super(MF_Polar_Align, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.embeddings = [self.user_emb, self.item_emb]

        # 全局bias
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

        self.sparse_var = torch.nn.Parameter(torch.ones((2, embedding_size)))

    def sparse_info(self, thres=0.001):
        with torch.no_grad():
            sparse_weight = torch.sigmoid(self.sparse_var * 15)
            sparse_ratio = 1 - torch.sum(sparse_weight > thres).item() / sparse_weight.view(-1).shape[0]
            remained_dim = torch.sum(sparse_weight > thres, dim=-1).tolist()
        return sparse_ratio, remained_dim

    def forward(self, fields, alpha=None):
        sparse_vec = torch.sigmoid(self.sparse_var * 15)

        u_id = fields[:,0]
        i_id = fields[:,1]
        U = self.user_emb(u_id) * sparse_vec[0]
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id) * sparse_vec[1]
        b_i = self.item_bias(i_id).squeeze()
        out = torch.sigmoid((U * I).sum(1) + b_u + b_i + self.mean)

        dist = torch.Tensor([0]).cuda()
        if alpha is not None:
            with torch.no_grad():
                if self.sim_dim_indices is None:
                    self.sim_dim_indices = self.similar_embedding_indices(alpha)

            slot, dim1, dim2 = self.sim_dim_indices
            if len(slot) > 0:
                dist = (trans_embed_x[:, slot, dim1] - trans_embed_x[:, slot, dim2]).square().mean()

        return out, dist

    def similar_embedding_indices(self, alpha):
        dists = []
        for i, field_emb in enumerate(self.embeddings):
            new_field_emb_weight = field_emb.weight.data
            dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
                               new_field_emb_weight.transpose(0, 1), p=2)#.squeeze()
            idx = torch.triu_indices(*dist.shape, offset=1)
            dists.append(dist[idx[0], idx[1]])
        dists = torch.cat(dists, dim=-1)
        thres = dists.kthvalue(max(1, int(dists.numel() * alpha)))[0]

        slot, dim1, dim2 = [], [], []
        i = 0
        for field_emb in self.embeddings:
            new_field_emb_weight = field_emb.weight.data
            if new_field_emb_weight.numel() == 0:
                continue
            # dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
            #                    new_field_emb_weight.transpose(0, 1), p=2)#.squeeze()
            dist = torch.from_numpy(np.corrcoef(new_field_emb_weight.transpose(0, 1).cpu().numpy()) * -1).cuda()
            dist_masked = dist.triu(diagonal=1) + torch.ones_like(dist).tril() * 1e5
            idx = (dist_masked < thres).nonzero()
            for (row_i, col_i) in idx.tolist():
                slot.append(i)
                dim1.append(row_i)
                dim2.append(col_i)
            i += 1

        slot = torch.Tensor(slot).long().cuda()
        dim1 = torch.Tensor(dim1).long().cuda()
        dim2 = torch.Tensor(dim2).long().cuda()

        return slot, dim1, dim2

    def cal_sim_groups(self, alpha=0.1):

        sparse_vec = torch.sigmoid(self.sparse_var * 15)
        sparse_mask = sparse_vec > 0.001
        sparse_vec = sparse_vec * sparse_mask

        dists = []
        for i, field_emb in enumerate(self.embeddings):
            non_zero_idxs = sparse_vec[i].nonzero().squeeze(-1)
            if non_zero_idxs.numel() <= 1:
                continue
            new_field_emb_weight = field_emb.weight.data[:, non_zero_idxs]
            # new_field_emb_weight = new_field_emb_weight * sparse_vec[i][non_zero_idxs]
            # new_field_emb_weight = field_emb.weight.data
            # dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
            #                    new_field_emb_weight.transpose(0, 1), p=2)#.squeeze()
            dist = torch.from_numpy(np.corrcoef(new_field_emb_weight.transpose(0, 1).cpu().numpy()) * -1).cuda()
            idx = torch.triu_indices(*dist.shape, offset=1)
            dists.append(dist[idx[0], idx[1]])
        dists = torch.cat(dists, dim=-1)
        thres = dists.kthvalue(max(1, int(dists.numel() * alpha)))[0]
        # print('thres: ', thres)

        field_idxs, dim_idxs = [], []
        for i, field_emb in enumerate(self.embeddings):
            non_zero_idxs = sparse_vec[i].nonzero().squeeze(-1)
            if non_zero_idxs.numel() <= 1:
                continue
            new_field_emb_weight = field_emb.weight.data[:, non_zero_idxs]
            # dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
            #                    new_field_emb_weight.transpose(0, 1), p=2)#.squeeze()
            dist = torch.from_numpy(np.corrcoef(new_field_emb_weight.transpose(0, 1).cpu().numpy()) * -1).cuda()
            dist_masked = dist.triu(diagonal=1) + torch.ones_like(dist).tril() * 1e5
            sim_idxs = (dist_masked < thres).nonzero().cpu().tolist()
            for idx in sim_idxs:
                field_idxs.append([i, i])
                dim_idxs.append([non_zero_idxs[idx[0]].item(), non_zero_idxs[idx[1]].item()])
                # sim_groups.append([[i, i], [non_zero_idxs[idx[0]].item(), non_zero_idxs[idx[1]].item()]])
            # sim_groups.append(sim_idxs)

        self.sim_groups = field_idxs, dim_idxs