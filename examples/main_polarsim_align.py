import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import random
import numpy as np
import ast


import os
import sys
sys.path.insert(0,'..')
from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel_SimPrune_Polar_Align
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel_SimPrune_Polar_Align
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel_SimPrune_Polar_Align
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork
from torchfm.model.mf import MF_Polar_Align

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()   

def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset, embed_dim, mlp_dims):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    # print(field_dims)
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel_SimPrune_Polar_Align(field_dims, embed_dim=embed_dim)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel_SimPrune_Polar_Align(field_dims, embed_dim=embed_dim, mlp_dims=mlp_dims, dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel_SimPrune_Polar_Align(field_dims, embed_dim=embed_dim, mlp_dims=mlp_dims, dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    elif name == 'mf':
        return MF_Polar_Align(field_dims[0], field_dims[1], 0, embed_dim)
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100, temper=1.4, alpha=1.0, lbd=5e-5, beta=0.1, sim=True, soft=True):
    model.train()
    total_loss, total_sim_emb_loss, total_sim_count = 0, 0, 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    if sim and beta > 0:
        model.cal_sim_groups(beta)
    import time
    model_time = 0
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y, _ = model(fields)
        bce_loss = criterion(y, target.float())
        sparse_weight = torch.sigmoid(model.sparse_var * 15)
        if soft:
            sparse_thres = torch.sigmoid(model.sparse_thres)
        else:
            sparse_thres = torch.mean(sparse_weight)
        all_sparse_term = temper * torch.sum(torch.abs(sparse_weight)) - torch.sum(torch.abs(sparse_weight - alpha * sparse_thres))
        loss = bce_loss + lbd * all_sparse_term
        if sim and beta > 0:
            if len(model.sim_groups[0]) > 0:
                g_sparse_weight = sparse_weight[model.sim_groups]
                g_sparse_mean = g_sparse_weight.mean(dim=1).unsqueeze(-1)
                sim_sparse_term = temper * torch.sum(torch.abs(g_sparse_weight), dim=1) - torch.sum(torch.abs(g_sparse_weight - alpha * g_sparse_mean), dim=1)
                sim_sparse_term = sim_sparse_term.mean()
                # sim_sparse_term = torch.Tensor([0]).to(device)
                # for g_index in model.sim_groups:
                #     # g_index = [[field_i, field_i], [dim1, dim2]]
                #     g_sparse_weight = sparse_weight[g_index]
                #     g_weight_mean = torch.mean(g_sparse_weight)
                #     sim_sparse_term += temper * torch.sum(torch.abs(g_sparse_weight)) - torch.sum(torch.abs(g_sparse_weight - alpha * g_weight_mean))
                loss = loss + lbd * 5 * sim_sparse_term
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            sparse_ratio, _ = model.sparse_info()
            tk0.set_postfix(loss=total_loss / log_interval,
                            all_sparse=all_sparse_term.item() * lbd,
                            # sim_sparse=sim_sparse_term.item() * lbd if sim and beta > 0 else 0,
                            sparse_ratio=sparse_ratio)
            total_loss, total_sim_count = 0, 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y, _ = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         early_stop_trials,
         embed_dim,
         mlp_dims,
         temper,
         alpha,
         beta,
         lbd,
         soft,
         pretrain_ckpt):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset, embed_dim, mlp_dims).to(device)
    if pretrain_ckpt is not None:
        state_dict = torch.load(pretrain_ckpt)
        model.load_state_dict(state_dict, strict=False)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=early_stop_trials, 
                            save_path=os.path.join(save_dir, model_name+"_best.pt"))
    # auc = test(model, test_data_loader, device)
    # print(f'test auc: {auc}')
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device, temper=temper, alpha=alpha, lbd=lbd, beta=beta,
              sim=epoch_i >= 0 and epoch % 5 == 0, soft=soft)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc, 'sparse ratio: ', model.sparse_info()[0])
        test_auc = test(model, test_data_loader, device)
        print(f'test auc: {test_auc}')
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

    # Save pretrained end model
    torch.save(model.state_dict(), os.path.join(save_dir, model_name+"_end.pt"))
    
    # Load the best model during pretraining saved by early_stopper:
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name+"_best.pt")))

    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M', choices=["movielens1M", "avazu", "criteo"])
    parser.add_argument('--dataset_path', default='data/ml-1m/ratings.dat', 
                                    choices=["data/criteo/train.txt", "data/avazu/train", "data/ml-1m/ratings.dat"])
    parser.add_argument('--model_name', default='dfm')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp_dir', default='logs/dfm_on_movielens1M_field_level/')
    parser.add_argument('--exp_name', default='pretrained')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--early_stop_trials', type=int, default=30)

    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--mlp_dims', type=str, default="(16, 16)")

    parser.add_argument('--temper', type=float, default=1.4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lbd', type=float, default=5e-5)
    parser.add_argument('--soft', action='store_true')
    parser.add_argument('--pretrain_ckpt', type=str, default=None)


    args = parser.parse_args()

    args.mlp_dims = ast.literal_eval(args.mlp_dims)

    save_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sys.stdout = Logger(os.path.join(args.exp_dir, args.exp_name, "log.txt"))
    torch.multiprocessing.set_start_method('spawn')

    # Set random seeds:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         save_dir,
         args.early_stop_trials,
         args.embed_dim,
         args.mlp_dims,
         args.temper,
         args.alpha,
         args.beta,
         args.lbd,
         args.soft,
         args.pretrain_ckpt)
