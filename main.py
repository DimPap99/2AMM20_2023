import pandas as pd
from helpers import read_csv, load_pickled_data, pickle_data
import os
import matplotlib.pyplot as plt 
import plotly
import chart_studio.plotly as py
import plotly.graph_objects as go
import os, sys
from helpers import bfs, get_class_pairs, choose_node_based_on_neighbors
import numpy as np
import networkx as nx

TIMESTEPS =[13]#, 42, 35, 32, 29, 22, 9 ,20]

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
timestep_dataframe = merge_feats_class.loc[merge_feats_class['Time step'].isin(TIMESTEPS)].reset_index(drop=True)
timestep_edges = txs_edgelist[txs_edgelist['txId1'].isin(timestep_dataframe['txId'])].reset_index(drop=True)


timestep_list = timestep_dataframe.values.tolist()

edge_list = timestep_edges.values.tolist()

from Graph import Graph, Node
from visualizations import visualize_graph_from_list, visualize_graph_df

graph_of_interest = Graph()
graph_of_interest.create_from_lists(timestep_list, edge_list)
print(graph_of_interest.get_total_nodes())

