import torch
import random
import numpy as np

def build_memory(config, dataloader_list, bound_list, dataset, model, device, local_rank):
    memory_index_list = []
    memory_strategy = config["memory"]["memory_strategy"]
    for i, dataloader in enumerate(dataloader_list):
        with torch.no_grad():
            memory_index = []
            if memory_strategy == "random":
                for data in dataloader:
                    meta_index = data["meta_index"]
                    memory_index += meta_index.int().detach().cpu().tolist()
                random.shuffle(memory_index)
                b = min(len(memory_index), bound_list[i])
                memory_index_list.append(memory_index[:b])
            elif memory_strategy == "icarl":
                features = []
                for data in dataloader:
                    meta_index = data["meta_index"]
                    memory_index += meta_index.int().detach().cpu().tolist()
                    if hasattr(model, "module"):
                        data = model.module.input_preprocess(data)
                    else:
                        data = model.input_preprocess(data)

                    data = {k: v.to(device) for k, v in data.items()}
                    if hasattr(model, 'module'):
                        outputs = model.module(data)
                        feats = outputs['reference_feature']
                    else:
                        outputs = model(data)
                        feats = outputs['reference_feature']
                    features.append(feats.detach().cpu().numpy())
                features = np.concatenate(features)
                b = min(len(memory_index), bound_list[i])
                selected_indexes = icarl_selection(features, b)
                memory_index = np.array(memory_index)
                memory_index_list.append(memory_index[selected_indexes].tolist())
            else:
                raise NotImplementedError
    dataset.update_memory(memory_index_list)

def icarl_selection(features, nb_examplars):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]