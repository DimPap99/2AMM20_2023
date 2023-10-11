import pandas as pd
from helpers import read_csv, load_pickled_data, pickle_data
import os
import matplotlib.pyplot as plt 
import plotly
import chart_studio.plotly as py
import plotly.graph_objects as go
import os, sys


TIMESTEP = 20
DATASET_BASE = "dataset"
PICKLED_TIMESTEPS = os.path.join("exports", "split_on_timestep.pkl")

txs_classes:pd.DataFrame = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_classes.csv"), ret_Dataframe=True)
txs_edgelist:pd.DataFrame = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_edgelist.csv"), ret_Dataframe=True)
txs_features:pd.DataFrame =  read_csv(os.path.join(DATASET_BASE, "elliptic_txs_features.csv"), ret_Dataframe=True)


merge_feats_class = pd.merge(txs_features, txs_classes, on='txId')
merge_feats_class.shape
print(f"Total Nodes df shape: {txs_features.shape}")
print(f"total edges df shape: {txs_edgelist.shape}")
#timestep_graphs = load_pickled_data(PICKLED_TIMESTEPS)
timestep_dataframe = merge_feats_class.loc[merge_feats_class['Time step'] == TIMESTEP].reset_index(drop=True)
timestep_edges = txs_edgelist[txs_edgelist['txId1'].isin(timestep_dataframe['txId'])].reset_index(drop=True)


timestep_list = timestep_dataframe.values.tolist()

edge_list = timestep_edges.values.tolist()

from Graph import Graph, Node
from visualizations import visualize_graph_from_list, visualize_graph_df

graph_of_interest = Graph()
graph_of_interest.create_from_lists(timestep_list, edge_list)
# print(graph_of_interest.nodes[62537081].label)

#check node with max children. (ONLY Children)
max_children = 0
node_of_interest:Node = None
from helpers import bfs, get_class_pairs, choose_node_based_on_neighbors
import numpy as np
neigh = 18
node_of_interest:Node = choose_node_based_on_neighbors(graph_of_interest, neigh, label=1)
if node_of_interest is not None:
    total_disjoint_graphs = []
    total_nodes = graph_of_interest.node_txId_set.copy()
    
    print(len(total_nodes))
# while len(total_nodes) > 0:
    #node_of_interest_txId = list(total_nodes)[0]
    # print(node_of_interest_txId)
    # print(graph_of_interest.nodes[node_of_interest_txId].total_neighbors())
    result = bfs(graph_of_interest, node_of_interest.txId, keep_direction=True)
    pairs = list(result[0])
    visited_nodes = list(result[1])
    #print(len(pairs), len(visited_nodes))
    nodes_n_classes = []
    p = []
    for n in visited_nodes:
        nodes_n_classes.append([n, graph_of_interest.nodes[n].label])
        p.append(graph_of_interest.nodes[n].label)
       
    df1 = pd.DataFrame(nodes_n_classes, columns=['txId', 'class'])
    # print(df1)
    #visualize_graph_df(df1, txs_edgelist, f"Graph in timestep {TIMESTEP} that includes the node {node_of_interest.txId} and {neigh} neighbors")
    # print(len(visited_nodes))
    #cls_pairs = list(get_class_pairs(graph_of_interest, pairs))
    #print(len(cls_pairs))
    # total_disjoint_graphs.append([pairs, cls_pairs])
    # total_nodes.difference_update(total_nodes, visited_nodes)
    # print(len(total_disjoint_graphs))
    #print(len(cls_pairs))
    visualize_graph_from_list(p, pairs, f"Graph in timestep {TIMESTEP} that includes the node {node_of_interest.txId} and {neigh} neighbors")

    
else:
    print("No such node")