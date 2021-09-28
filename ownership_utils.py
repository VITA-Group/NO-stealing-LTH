
from hashlib import new
from networkx.algorithms.centrality.betweenness import edge_betweenness_centrality
import copy
import torch
import networkx
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from dataset import *

def need_to_prune(name, m, conv1):
    return ((name == 'conv1' and conv1) or (name != 'conv1')) \
        and isinstance(m, nn.Conv2d)
 
def calculate_betweenness(model, mask_dict, num_paths, args):
    new_mask_dict = copy.deepcopy(mask_dict)
    graphs = []
    graph = networkx.Graph()
    name_list = []

    for name,m in model.named_modules():
        if need_to_prune(name, m, args.conv1):
            name_list.append(name)

    for name,m in model.named_modules():
        graph = networkx.Graph()
        if need_to_prune(name, m, args.conv1):
            mask = mask_dict[name+'.weight_mask']
            weight = mask * m.weight
            weight = torch.sum(weight.abs(), [2, 3])
            for i in range(weight.shape[1]):
                start_name = name + '.{}'.format(i)
                graph.add_node(start_name)
                for j in range(weight.shape[0]):
                    try:
                        end_name = name_list[name_list.index(name) + 1] + '.{}'.format(j)
                        graph.add_node(end_name)
                        
                    except:
                        end_name = 'final.{}'.format(j)
                        graph.add_node(end_name)

                    graph.add_edge(start_name, end_name, weight=weight[j, i])
        graphs.append(graph)
    
    bs = []
    for graph in graphs:
        bs.append(edge_betweenness_centrality(graph))
    #print(bs)
    #assert False
    return bs
