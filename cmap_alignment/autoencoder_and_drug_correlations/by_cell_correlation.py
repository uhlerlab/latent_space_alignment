import sys
sys.path.insert(0, '../')
from configs.blanco_configs import  L1000_DATA_DIR
import numpy as np
from copy import deepcopy
import data_loader as dl
import options_parser as op
import random
import pickle as p
from sklearn.cluster import KMeans
import torch
import neural_model as nm
from torch.autograd import Variable


def get_embedding(net, x):
    o = Variable(deepcopy(x))
    for idx, layer in enumerate(net.net):
        o = layer(o)
        if idx == 0:
            break
    return o


def norm(vec):
    return np.sqrt(np.sum(np.power(vec,2)))


def get_correlation(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# The code below requires all datasets from
# https://github.com/uhlerlab/covid19_repurposing/tree/main/Code_autoencoding
def main(args):
    SEED = 1717
    np.random.seed(SEED)
    random.seed(SEED)
    autoencoder_flag = args.autoencoder_flag

    if autoencoder_flag:
        net = nm.Net()
        # Load a saved trained autoencoder model
        path = './trained_model_best.pth'
        d = torch.load(path)
        net.load_state_dict(d['state_dict'])
        net.eval()

    pairs = dl.get_all_pairs(args.data)
    cell_types = {}
    idx_to_cell = []
    cell_1 = 'A549'
    cell_2 = 'MCF7'
    fda_approved = set([])

    with open(L1000_DATA_DIR + 'fda_approved.txt', 'r') as f:
        for line in f:
            fda_approved.add(line.strip())

    pert_cell_map = p.load(open(L1000_DATA_DIR + 'pert_cell_map.p', 'rb'))
    pert_dose_map = p.load(open(L1000_DATA_DIR + 'pert_dose_map.p', 'rb'))
    embedding_map = {}
    drug_list = set([])

    for i in range(len(pairs)):
        cell_type = pert_cell_map[pairs[i][0]]
        pert_id = pert_dose_map[pairs[i][0]][2]
        pert_dose = pert_dose_map[pairs[i][0]][0]

        if cell_type != cell_1 and cell_type != cell_2:
            continue
        if pert_id not in fda_approved and pert_id != 'DMSO':
            continue

        cell_type = cell_type + "_" + pert_id
        drug_list.add(pert_id)

        idx_to_cell.append(cell_type)
        if cell_type in cell_types:
            cell_types[cell_type].append(i)
        else:
            cell_types[cell_type] = [i]

        embedding = pairs[i][1].reshape(1, -1)
        embedding = np.log2(embedding+1)
        embedding = (embedding - embedding.min()) \
                    / (embedding.max() - embedding.min())

        if autoencoder_flag:
            embedding = torch.from_numpy(embedding).view(1, -1).float()
            embedding = get_embedding(net, embedding).data.numpy()
        embedding_map[i] = embedding

    print(len(sorted(cell_types.keys())))
    control_embeddings = []
    for index in cell_types['A549_DMSO']:
        control_embeddings.append(embedding_map[index])
    control_indices = cell_types['A549_DMSO']
    control_embeddings = np.concatenate(control_embeddings,
                                        axis=0)
    print(control_embeddings.shape)
    kmeans = KMeans(n_clusters=2, random_state=SEED)
    kmeans = kmeans.fit(control_embeddings)
    labels = kmeans.labels_
    if sum(labels) > len(labels) // 2:
        selection = 1.
    else:
        selection = 0.
    new_indices = []
    for idx, label in enumerate(labels):
        if label == selection:
            new_indices.append(control_indices[idx])
    print("Number of A549 control used: ", len(new_indices))
    cell_types['A549_DMSO'] = new_indices

    control_embeddings = []
    for index in cell_types['MCF7_DMSO']:
        control_embeddings.append(embedding_map[index])
    control_indices = cell_types['MCF7_DMSO']
    control_embeddings = np.concatenate(control_embeddings,
                                        axis=0)
    print(control_embeddings.shape)
    kmeans = KMeans(n_clusters=2, random_state=SEED)
    kmeans = kmeans.fit(control_embeddings)
    labels = kmeans.labels_
    if sum(labels) > len(labels) // 2:
        selection = 1.
    else:
        selection = 0.
    new_indices = []
    for idx, label in enumerate(labels):
        if label == selection:
            new_indices.append(control_indices[idx])
    print("Number of MCF7 control used: ", len(new_indices))
    cell_types['MCF7_DMSO'] = new_indices

    means = {}
    for key in cell_types:
        points = np.array([embedding_map[i] for i in cell_types[key]])
        means[key] = np.mean(points, axis=0).reshape(-1)

    correlations = []
    drug_names = []

    for idx, d in enumerate(drug_list):
        if d != 'DMSO':
            key_1 = cell_1 + '_' + d
            if key_1 not in means:
                continue
            c = get_correlation(means[cell_1 + '_' + d] \
                                - means[cell_1 + '_DMSO'],
                                means[cell_2 + '_' + d] \
                                -means[cell_2 + '_DMSO'])
            correlations.append(c)
            drug_names.append(d)
    drug_corr_pairs = list(zip(drug_names, correlations))
    drug_pairs = sorted(drug_corr_pairs, reverse=True,
                        key = lambda k: k[1])

    with open('drug_correlations/by_cell_correlations.txt', 'w') as f:
        for pair in drug_pairs:
            f.write(str(pair) + '\n')


if __name__ == "__main__":
    args = op.setup_options()
    main(args)
