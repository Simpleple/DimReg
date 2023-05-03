import torch
import torch.nn.utils.prune as prune
import copy
from pruning_random_search import test
from tqdm import tqdm

def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Embedding):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False,
                            embedding_use_mask=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Embedding):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=embedding_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


# def prune_embedding_size_random_search_feature_level(model, device, valid_data_loader, 
#                                     prune_amount_emb_size, search_iterations):
#     auc_list = []
#     best_valid_auc = 0.0
#     best_model = None

#     for i in tqdm(range(0, search_iterations)):
#         # Prune a copy of the pretrained model
#         pruned_model = copy.deepcopy(model)
#         module = pruned_model.embedding.embedding

#         prune.random_unstructured(module, name="weight", amount=prune_amount_emb_size) # Pruning the embedding dimension at feature (weight) level
#         prune.remove(module, 'weight')

#         # Validate the pruned model
#         valid_auc = test(pruned_model, valid_data_loader, device)
#         auc_list.append(valid_auc)
#         if valid_auc >= best_valid_auc:
#             best_valid_auc = valid_auc
#             best_model = pruned_model
    
#     # measure sparsity after pruning
#     module = best_model.embedding.embedding
#     num_zeros, num_elements, sparsity = measure_module_sparsity(module, weight=True, bias=False, use_mask=False)
#     print("Sparsity in best_model.embedding.embedding.weight: {:.2f}%".format(sparsity))
#     num_zeros, num_elements, sparsity = measure_global_sparsity(best_model, weight=True, bias=False, 
#                 conv2d_use_mask=False, linear_use_mask=False, embedding_use_mask=False)
#     print("Global Sparsity in best_model: {:.2f}%".format(sparsity))
    
#     return auc_list, best_valid_auc, best_model


def prune_embedding_size_random_field_level(model, prune_amount_emb_size):
    modules = model.embedding.embeddings
    for module in modules:
        prune.random_structured(module, name="weight", amount=prune_amount_emb_size, dim=1)
        prune.remove(module, 'weight')
    return model

def prune_embedding_size_random_search(model, device, valid_subset_data_loader, 
                                    prune_amount_emb_size, search_iterations):
    auc_list = []
    best_valid_auc = 0.0
    best_model = None

    for i in tqdm(range(0, search_iterations)):
        # Prune a copy of the pretrained model
        pruned_model = copy.deepcopy(model)
        modules = pruned_model.embedding.embeddings

        # Prune each field separately
        for module in modules:
            #  Prune embedding dimension at the field level
            prune.random_structured(module, name="weight", amount=prune_amount_emb_size, dim=1) # dim=1 is the dimension axis
            prune.remove(module, 'weight')

        # Validate the pruned model
        valid_auc = test(pruned_model, valid_subset_data_loader, device)
        auc_list.append(valid_auc)
        if valid_auc >= best_valid_auc:
            best_valid_auc = valid_auc
            best_model = pruned_model
    
    # measure sparsity after pruning
    modules = best_model.embedding.embeddings
    for idx, module in enumerate(modules):
        num_zeros, num_elements, sparsity = measure_module_sparsity(module, weight=True, bias=False, use_mask=False)
        print("Sparsity in best_model.embedding.embeddings.{:d}.weight: {:.2f}%".format(idx, sparsity))    
    num_zeros, num_elements, sparsity = measure_global_sparsity(best_model, weight=True, bias=False, 
                conv2d_use_mask=False, linear_use_mask=False, embedding_use_mask=False)
    print("Global Sparsity in best_model: {:.2f}%".format(sparsity))
    
    return auc_list, best_valid_auc, best_model
