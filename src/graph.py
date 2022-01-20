import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

from .data import Dictionary

def load_triplets(ent_rel_dir):
    logger.info("load data from {}".format(ent_rel_dir))
    load_relation2idx = os.path.join(ent_rel_dir,"relation2idx.json")
    load_entity2idx = os.path.join(ent_rel_dir,"entity2idx.json")
    load_head2relation2tail = os.path.join(ent_rel_dir,"head2relation2tail.csv")
    ent_dict = Dictionary.load(load_entity2idx)
    rel_dict = Dictionary.load(load_relation2idx)
    
    all_triplets = pd.read_csv(load_head2relation2tail)

    logger.info('num_entity: {}'.format(len(ent_dict)))
    logger.info('num_relation: {}'.format(len(rel_dict)))
    logger.info('num_all_triples: {}'.format(len(all_triplets)))

    return ent_dict, rel_dict, all_triplets.values
def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]
    return edge_norm
def build_triplets_graph(num_nodes,num_rels,data_triplets,percentage=0.7):
    src, rel, dst = data_triplets.transpose()

    src = torch.from_numpy(src)
    rel = torch.from_numpy(rel)
    dst = torch.from_numpy(dst)
    
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    
    rel = torch.cat((rel, rel))
    
    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)
    dataset_len = edge_index.shape[1]
    train_len = int(dataset_len*percentage)
    all_set = np.arange(0,train_len)
    np.random.shuffle(all_set)
    data.train_mask = torch.zeros((dataset_len,),dtype=torch.long)
    data.train_mask[torch.from_numpy(all_set[:train_len])] = 1
    data.test_mask = torch.zeros((dataset_len,),dtype=torch.long)
    data.test_mask[torch.from_numpy(all_set[train_len:])] = 1
    return data
