from ast import Return
import numpy as np
import torch
import torch.nn.functional as F


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


# class FeaturesEmbeddingFeatureLevel(torch.nn.Module):

#     def __init__(self, field_dims, embed_dim):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
#         self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
#         torch.nn.init.xavier_uniform_(self.embedding.weight.data)

#     def forward(self, x):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         if not self.pruned:
#             """
#             Old attributes before pruning:
#             - embedding
#             - offsets
#             """
#             x = x + x.new_tensor(self.offsets).unsqueeze(0)
#             return self.embedding(x)
#         else:
#             """
#             New attributes after pruning:
#             - new_field_dims
#             - new_offsets
#             - embed_dim
#             - transform_matrices
#             - feature_embeddings
#             - pruned
#             """
#             x = x + x.new_tensor(self.offsets).unsqueeze(0)

#             outputs = list()
#             for instance_i in range(x.shape[0]):
#                 embs = list()
#                 for field_i in range(x.shape[-1]):
#                     feature_i = x[instance_i, field_i].item()
#                     emb_layer = self.feature_embeddings[str((field_i, feature_i))]
#                     if emb_layer is None:
#                         embs.append(torch.zeros((self.embed_dim)).to(x.device))
#                     else:
#                         emb = emb_layer(torch.tensor([0], dtype=torch.long).to(x.device))
#                         # Transform pruned embedding to the same dimension
#                         emb = torch.matmul(self.transform_matrices[str((field_i, emb.shape[-1]))], emb.squeeze(0))
#                         embs.append(emb)
#                 embs = torch.stack(embs, dim=0)
#                 outputs.append(embs)
#             outputs = torch.stack(outputs, dim=0)

#             return outputs

#     def prune(self, field_dims, embed_dim, reinitialize):
#         """
#         Description:
#             remove embedding parameters with 0 value, and use additonal parameter 
#             matrix to transform each field back to consitent shape
#         Input:
#             - field_dims: a list of original input field dimensions
#             - embed_dim: embedding hidden dimension
#         """
#         if self.pruned:
#             return

#         # Save pretrained weights
#         pretrained_weights = self.embedding.weight.data.cpu().numpy()
#         # Index of non-zero weights
#         idxs_to_keep = np.where(pretrained_weights==0, False, True)
#         # Find the pruned dimension of each embedding
#         pruned_embed_dims = np.count_nonzero(idxs_to_keep, axis=1)

#         # Find the pruned new_field_dims
#         self.new_field_dims = []
#         field_offsets = np.array((0, *np.cumsum(field_dims)), dtype=np.long)
#         for field_i in range(len(field_offsets)-1):
#             # Count how many feature in current field are not totaly pruned (with dimension size>0)
#             self.new_field_dims.append(np.count_nonzero(
#                         pruned_embed_dims[field_offsets[field_i]:field_offsets[field_i+1]]))
#         # Find the pruned new_offsets
#         self.new_offsets = np.array((0, *np.cumsum(self.new_field_dims)[:-1]), dtype=np.long)

#         # Create transformation matrices for each field separately.
#         # self.d_max = max(list(set(pruned_embed_dims))) # maximum dimension among all features embeddings
#         self.embed_dim = embed_dim
#         self.transform_matrices = torch.nn.ParameterDict()
#         for field_i in range(len(field_offsets)-1):
#             unique_embed_dims = list(set(pruned_embed_dims[field_offsets[field_i]:field_offsets[field_i+1]]))
#             # For each embedding dimension of each field
#             for current_embed_dim in unique_embed_dims:
#                 self.transform_matrices[str((field_i, current_embed_dim))] = torch.nn.Parameter(torch.zeros((self.embed_dim, current_embed_dim)))
#                 torch.nn.init.xavier_uniform_(self.transform_matrices[str((field_i, current_embed_dim))].data)
        
#         # Create New Pruned Embedding Layers for each field
#         self.feature_embeddings = torch.nn.ModuleDict([])
#         for field_i in range(len(field_offsets)-1):
#             for feature_i in range(field_offsets[field_i], field_offsets[field_i+1]):
#                 # Current Feature Field is Totally Pruned
#                 if pruned_embed_dims[feature_i] == 0:
#                     self.feature_embeddings[str((field_i, feature_i))] = None # Fix the pruned embedding to constant zeros of shape (self.embed_dim)
#                 else:
#                     self.feature_embeddings[str((field_i, feature_i))] = torch.nn.Embedding(1, pruned_embed_dims[feature_i])
#                     # Reinitialize weigth or inherent pre-trained weights
#                     if reinitialize:
#                         torch.nn.init.xavier_uniform_(self.feature_embeddings[str((field_i, feature_i))].weight.data)
#                     else:
#                         self.feature_embeddings[str((field_i, feature_i))].weight.data.copy_(
#                                 torch.from_numpy(pretrained_weights[feature_i][idxs_to_keep[feature_i]]))

#         self.pruned = True

#         return

class FeaturesEmbeddingVariedLength(torch.nn.Module):

    def __init__(self, field_dims, embed_dims):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(field_dim, embed_dim)
                                               for (field_dim, embed_dim) in zip(field_dims, embed_dims)])
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embs = list()

        for field_i in range(self.num_fields):
            emb_layer = self.embeddings[field_i]
            emb = emb_layer(x[:, field_i])
            embs.append(emb)

        # embs = torch.cat(embs, dim=1)
        return embs


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(field_dim, embed_dim) for field_dim in field_dims])
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
        self.pruned = False

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embs = list()

        for field_i in range(self.num_fields):
            emb_layer = self.embeddings[field_i]
            
            if not self.pruned: # Original model
                """
                Old attributes before pruning:
                - num_fields
                - embeddings
                """
                emb = emb_layer(x[:,field_i])
            else: # Pruned model
                """
                New attributes after pruning:
                - embed_dim
                - transform_matrices
                - embeddings
                - pruned
                """
                if emb_layer is None:
                    emb = torch.zeros((self.embed_dim)).to(x.device)
                else:
                    emb = emb_layer(x[:,field_i])
                    emb = torch.matmul(self.transform_matrices[field_i], emb.t()).t()
            embs.append(emb)

        embs = torch.stack(embs, dim=1)
        return embs

    def prune(self, field_dims, embed_dim, reinitialize):
        """
        Description:
            remove embedding parameters with 0 value, and use additonal parameter 
            matrix to transform each field back to consitent shape
        Input:
            - field_dims: a list of original input field dimensions
            - embed_dim: embedding hidden dimension
        """
        # Implement Remove parameters for field level embedding dimension pruning.
        if self.pruned:
            return

        # Save pretrained weights
        pretrained_weights = [embedding.weight.data.cpu().numpy() for embedding in self.embeddings]

        # Index of non-zero weights
        idxs_to_keep = [np.where(pretrained_weight==0, False, True) for pretrained_weight in pretrained_weights]
        # Set each field weight masks for copying weight, set entire column to zero/one if any/no zero exist
        idxs_to_keep = [np.count_nonzero(idxs, axis=0) != 0 for idxs in idxs_to_keep]
        idxs_to_keep = [np.tile(idxs_to_keep[field_i], (field_dims[field_i], 1)) for field_i in range(len(pretrained_weights))]

        # Find the pruned dimension of each embedding
        pruned_embed_dims = [np.count_nonzero(np.count_nonzero(idxs, axis=0) != 0) for idxs in idxs_to_keep]

        # Create transformation matrices for each field separately.
        self.embed_dim = embed_dim
        self.transform_matrices = torch.nn.ParameterList()
        for field_i in range(len(pretrained_weights)):
            self.transform_matrices.append(torch.nn.Parameter(torch.zeros((self.embed_dim, pruned_embed_dims[field_i]))))
            torch.nn.init.xavier_uniform_(self.transform_matrices[field_i].data)
        
        # Create New Pruned Embedding Layers for each field
        self.embeddings = torch.nn.ModuleList([])
        for field_i in range(len(pretrained_weights)):
            # If Current Field Dimension is Totally Pruned
            if pruned_embed_dims[field_i] == 0:
                self.embeddings.append(None) # Fix the pruned embedding to constant zeros of shape (self.embed_dim)
            else:
                self.embeddings.append(torch.nn.Embedding(field_dims[field_i], pruned_embed_dims[field_i]))
                # Reinitialize weight or inherent pre-trained weights
                if reinitialize:
                    torch.nn.init.xavier_uniform_(self.embeddings[field_i].weight.data)
                else:
                    temp_weight = pretrained_weights[field_i][idxs_to_keep[field_i]].reshape(self.embeddings[field_i].weight.data.shape)
                    self.embeddings[field_i].weight.data.copy_(
                            torch.from_numpy(temp_weight))

        self.pruned = True

        return


class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = F.dropout(attn_output, p=self.dropouts[1], training=self.training)
        return self.fc(attn_output)


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class AnovaKernel(torch.nn.Module):

    def __init__(self, order, reduce_sum=True):
        super().__init__()
        self.order = order
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        batch_size, num_fields, embed_dim = x.shape
        a_prev = torch.ones((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
        for t in range(self.order):
            a = torch.zeros((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
            a[:, t+1:, :] += x[:, t:, :] * a_prev[:, t:-1, :]
            a = torch.cumsum(a, dim=1)
            a_prev = a
        if self.reduce_sum:
            return torch.sum(a[:, -1, :], dim=-1, keepdim=True)
        else:
            return a[:, -1, :]
