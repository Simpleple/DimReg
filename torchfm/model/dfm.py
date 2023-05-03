import numpy as np
import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear\
    , MultiLayerPerceptron, FeaturesEmbeddingVariedLength


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class DeepFactorizationMachineModel_PolarSim_Align(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.transform = torch.nn.ModuleList([torch.nn.Linear(embed_dim, embed_dim) for _ in field_dims])

        self.sparse_var = torch.nn.Parameter(torch.ones((len(field_dims), embed_dim)))
        self.sparse_thres = torch.nn.Parameter(torch.ones((len(field_dims), embed_dim)))
        self.pruned = False

    def sparse_info(self, thres=0.001):
        with torch.no_grad():
            sparse_weight = torch.sigmoid(self.sparse_var * 15)
            sparse_ratio = 1 - torch.sum(sparse_weight > thres).item() / sparse_weight.view(-1).shape[0]
            remained_dim = torch.sum(sparse_weight > thres, dim=-1).tolist()
        return sparse_ratio, remained_dim

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        sparse_vec = torch.sigmoid(self.sparse_var.detach() * 15)
        if not self.training:
            sparse_mask = sparse_vec > 0.001
            sparse_vec = sparse_mask.float() * sparse_vec
        if self.pruned:
            new_embed_x = self.pruned_embedding(x)# * sparse_vec
            trans_embed_x = torch.stack([fc(e) for fc, e in zip(self.transform, new_embed_x)], dim=1)
            x = self.linear(x) + self.fm(trans_embed_x) + self.mlp(
                torch.cat(new_embed_x, dim=1).view(-1, self.mlp.mlp[0].in_features))
        else:
            embed_x = self.embedding(x) * sparse_vec
            trans_embed_x = torch.stack([self.transform[i](embed_x[:,i,:]) for i in range(len(self.transform))
                                         if embed_x[:,i,:].numel() > 0], dim=1)
            x = self.linear(x) + self.fm(trans_embed_x) + self.mlp(embed_x.view(-1, self.mlp.mlp[0].in_features))

        return torch.sigmoid(x.squeeze(1))

    def cal_sim_groups(self, alpha=0.1):

        sparse_vec = torch.sigmoid(self.sparse_var * 15)
        sparse_mask = sparse_vec > 0.001
        sparse_vec = sparse_vec * sparse_mask

        dists = []
        for i, field_emb in enumerate(self.embedding.embeddings):
            non_zero_idxs = sparse_vec[i].nonzero().squeeze(-1)
            if non_zero_idxs.numel() == 0:
                continue
            new_field_emb_weight = field_emb.weight.data[:, non_zero_idxs]
            # new_field_emb_weight = new_field_emb_weight * sparse_vec[i][non_zero_idxs]
            # new_field_emb_weight = field_emb.weight.data
            dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
                               new_field_emb_weight.transpose(0, 1), p=2)#.squeeze()
            idx = torch.triu_indices(*dist.shape, offset=1)
            dists.append(dist[idx[0], idx[1]])
        dists = torch.cat(dists, dim=-1)
        thres = dists.kthvalue(max(1, int(dists.numel() * alpha)))[0]
        # print('thres: ', thres)

        sim_groups = []
        for i, field_emb in enumerate(self.embedding.embeddings):
            non_zero_idxs = sparse_vec[i].nonzero().squeeze(-1)
            if non_zero_idxs.numel() == 0:
                continue
            new_field_emb_weight = field_emb.weight.data[:, non_zero_idxs]
            dist = torch.cdist(new_field_emb_weight.transpose(0, 1),
                               new_field_emb_weight.transpose(0, 1), p=2)#.squeeze()
            dist_masked = dist.triu(diagonal=1) + torch.ones_like(dist).tril() * 1e5
            sim_idxs = (dist_masked < thres).nonzero().cpu().tolist()
            for idx in sim_idxs:
                sim_groups.append([[i, i], [non_zero_idxs[idx[0]], non_zero_idxs[idx[1]]]])
                # sim_groups.append([i] + [non_zero_idxs[j].item() for j in idx])
            # sim_groups.append(sim_idxs)

        self.sim_groups = sim_groups

