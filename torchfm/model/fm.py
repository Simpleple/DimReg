import torch
import numpy as np
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, FeaturesEmbeddingVariedLength


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


class FactorizationMachineModel_SimPrune_Polar_Align(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

        self.transform = torch.nn.ModuleList([torch.nn.Linear(embed_dim, embed_dim) for _ in field_dims])

        self.sparse_var = torch.nn.Parameter(torch.ones((len(field_dims), embed_dim)))
        self.sparse_thres = torch.nn.Parameter(torch.ones((len(field_dims), embed_dim)))
        self.pruned = False
        self.sim_dim_indices = None

    def sparse_info(self, thres=0.001):
        with torch.no_grad():
            sparse_weight = torch.sigmoid(self.sparse_var * 15)
            sparse_ratio = 1 - torch.sum(sparse_weight > thres).item() / sparse_weight.view(-1).shape[0]
            remained_dim = torch.sum(sparse_weight > thres, dim=-1).tolist()
        return sparse_ratio, remained_dim

    def forward(self, x, alpha=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        sparse_vec = torch.sigmoid(self.sparse_var * 15)
        if not self.training:
            sparse_mask = sparse_vec > 0.001
            sparse_vec = sparse_mask.float() * sparse_vec
        new_embed_x = self.embedding(x) #* sparse_vec
        if self.pruned:
            trans_embed_x = torch.stack([self.transform[i](new_embed_x[i]) if new_embed_x[i].numel() > 0
                                         else self.transform[i].bias.data.detach().clone().repeat(x.shape[0], 1)
                                         for i in range(len(self.transform))], dim=1)
            x = self.linear(x) + self.fm(trans_embed_x)
        else:
            new_embed_x = new_embed_x * sparse_vec
            trans_embed_x = torch.stack([self.transform[i](new_embed_x[:,i,:]) for i in range(len(self.transform))
                                         if new_embed_x[:,i,:].numel() > 0], dim=1)
            x = self.linear(x) + self.fm(trans_embed_x)

        dist = torch.Tensor([0]).cuda()
        if alpha is not None:
            with torch.no_grad():
                if self.sim_dim_indices is None:
                    self.sim_dim_indices = self.similar_embedding_indices(alpha)

            slot, dim1, dim2 = self.sim_dim_indices
            if len(slot) > 0:
                dist = (trans_embed_x[:, slot, dim1] - trans_embed_x[:, slot, dim2]).square().mean()

        return torch.sigmoid(x.squeeze(1)), dist

    def similar_embedding_indices(self, alpha):
        dists = []
        for i, field_emb in enumerate(self.embedding.embeddings):
            new_field_emb_weight = field_emb.weight.data
            dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
                               new_field_emb_weight.transpose(0, 1), p=2)#.squeeze()
            idx = torch.triu_indices(*dist.shape, offset=1)
            dists.append(dist[idx[0], idx[1]])
        dists = torch.cat(dists, dim=-1)
        thres = dists.kthvalue(max(1, int(dists.numel() * alpha)))[0]

        slot, dim1, dim2 = [], [], []
        i = 0
        for field_emb in self.embedding.embeddings:
            new_field_emb_weight = field_emb.weight.data
            if new_field_emb_weight.numel() == 0:
                continue
            dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
                               new_field_emb_weight.transpose(0, 1), p=2)#.squeeze()
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

    def polar_prune(self):
        sparse_vec = torch.sigmoid(self.sparse_var * 15)
        sparse_mask = sparse_vec > 0.001
        sparse_vec = sparse_vec * sparse_mask

        self.pruned = True

        # embedding prune
        num_embeddings = [embed.num_embeddings for embed in self.embedding.embeddings]
        field_dims = sparse_mask.sum(dim=1).cpu().detach().tolist()
        pruned_embedding = FeaturesEmbeddingVariedLength(num_embeddings, field_dims)
        for i, field_emb in enumerate(self.embedding.embeddings):
            new_emb_weight = field_emb.weight.data.detach() * sparse_vec[i]
            pruned_embedding.embeddings[i].weight.data = new_emb_weight[:, sparse_mask[i]]

        self.embedding = pruned_embedding

        # transform layer prune
        new_layers = []
        for i in range(len(self.embedding.embeddings)):
            if sparse_mask[i].sum() != 0:
                new_transform_weight = self.transform[i].weight.data.detach()[:, sparse_mask[i]]
                new_trans_layer = torch.nn.Linear(new_transform_weight.shape[1], new_transform_weight.shape[0])
                new_trans_layer.weight.data = new_transform_weight
                new_trans_layer.bias.data = self.transform[i].bias.data
                new_layers.append(new_trans_layer)
            else:
                new_trans_layer = torch.nn.Linear(1, sparse_mask.shape[1])
                new_trans_layer.bias.data = self.transform[i].bias.data
                new_layers.append(new_trans_layer)
                # new_layers.append(self.transform[i])
            # self.transform[i] = new_trans_layer
        self.transform = torch.nn.ModuleList(new_layers)

        print('pruned embeddings: ', sum(field_dims))

    def sim_prune(self, alpha=0.1):
        num_embeddings = [embed.num_embeddings for embed in self.embedding.embeddings]
        field_dims = [embed.embedding_dim for embed in self.embedding.embeddings]
        field_dim_offset = np.cumsum([0] + field_dims)

        pruned_embedding = FeaturesEmbeddingVariedLength(num_embeddings, field_dims)

        # dists = []
        # for i, field_emb in enumerate(self.embedding.embeddings):
        #     new_field_emb_weight = field_emb.weight.data
        #     dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
        #                        new_field_emb_weight.transpose(0, 1), p=2)#.squeeze()
        #     idx = torch.triu_indices(*dist.shape, offset=1)
        #     dists.append(dist[idx[0], idx[1]])
        # dists = torch.cat(dists, dim=-1)
        # thres = dists.kthvalue(max(1, int(dists.numel() * alpha)))[0]
        thres = alpha
        print('thres: ', thres)

        for i, field_emb in enumerate(self.embedding.embeddings):

            new_field_emb_weight = field_emb.weight.data.clone().detach()

            new_transform_weight = self.transform[i].weight.data.clone().detach()
            while new_field_emb_weight.shape[1] > 1:
                max_num_embeddings = 100000 # avoid cuda out of memory
                if new_field_emb_weight.shape[0] > max_num_embeddings and new_field_emb_weight.shape[1] > 10:
                    half_dim = new_field_emb_weight.shape[1] // 2
                    dist1 = torch.cdist(new_field_emb_weight[:, :half_dim].transpose(0, 1),
                                        new_field_emb_weight.transpose(0, 1), p=2).squeeze()
                    dist2 = torch.cdist(new_field_emb_weight[:, half_dim:].transpose(0, 1),
                                        new_field_emb_weight.transpose(0, 1), p=2).squeeze()
                    dist = torch.cat([dist1, dist2], dim=0)
                else:
                    dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
                                       new_field_emb_weight.transpose(0, 1), p=2).squeeze()
                dist_masked = dist.triu(diagonal=1) + torch.ones_like(dist).tril() * 1e5
                min_dist = dist_masked.min()
                if min_dist >= thres:
                    break
                idx = (dist_masked == min_dist).nonzero()[0]

                avg_emb = new_field_emb_weight[:, idx[:2]].mean(dim=1)
                new_field_emb_weight[:, idx[0]] = avg_emb
                new_field_emb_weight = new_field_emb_weight[:, np.arange(new_field_emb_weight.shape[1]) != idx[1].item()]

                sum_trans_weight = new_transform_weight[:, idx[:2]].sum(dim=1)
                new_transform_weight[:, idx[0]] = sum_trans_weight
                new_transform_weight = new_transform_weight[:, np.arange(new_transform_weight.shape[1]) != idx[1].item()]

            new_embedding = torch.nn.Embedding(num_embeddings[i], new_field_emb_weight.shape[1])
            new_embedding.weight.data = new_field_emb_weight
            pruned_embedding.embeddings[i] = new_embedding
            # self.pruned_embedding.embeddings[i] = field_emb
            # print(f'{i}: {new_field_emb_weight.shape[1]}')

            new_trans_layer = torch.nn.Linear(new_transform_weight.shape[1], new_transform_weight.shape[0])
            new_trans_layer.weight.data = new_transform_weight
            new_trans_layer.bias.data = self.transform[i].bias.data
            self.transform[i] = new_trans_layer

        self.embedding = pruned_embedding

        new_field_dims = [embed.embedding_dim for embed in self.embedding.embeddings]
        print(f"dim before prune: {sum(field_dims)}, "
              f"dim after prune: {sum(new_field_dims)}, "
              f"prune ratio: {1 - sum(new_field_dims) / sum(field_dims)}")

        self.sim_dim_indices = None
        self.pruned = True

    def cal_sim_groups(self, alpha=0.1):

        sparse_vec = torch.sigmoid(self.sparse_var * 15)
        sparse_mask = sparse_vec > 0.001
        sparse_vec = sparse_vec * sparse_mask

        dists = []
        for i, field_emb in enumerate(self.embedding.embeddings):
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
        for i, field_emb in enumerate(self.embedding.embeddings):
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