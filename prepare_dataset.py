from helpers import read_csv, export_csv
from Graph import Graph, Node
import os

DATASET_BASE = "dataset"

#Prepare two datasets. 1 with all the connected nodes. 1 with all the connected nodes that belong to either class 1 or 2
def join_information_accross_datasets():
    txs_classes = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_classes.csv"))
    txs_edgelist = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_edgelist.csv"))
    txs_features =  read_csv(os.path.join(DATASET_BASE, "elliptic_txs_features.csv"), drop_header=True)
   
    split_on_timestep = {}
    graph = Graph()
    #initialize nodes with features
    for row in txs_features:
        graph.add_Node(Node(feats=row))
    
    #assign class to nodes
    for row in txs_classes:
        txId = row[0]
        label = 3 if row[1] == "Uknown" else row[1]
        if txId in graph.nodes:
            graph.nodes[txId].label = label
    
    #find neighboring nodes
    for row in txs_edgelist:
        current_txId = row[0]
        neighboring_txId = row[1]
        if current_txId in graph.nodes and neighboring_txId in graph.nodes:
            graph.add_neighbor_to_node(current_txId, neighboring_txId)
    num_of_neighbors = dict()
    zero_neighbors = 0
    for k, v in graph.nodes.items():
        node:Node = v
        n_of_neighbors = node.get_num_of_neighbors()
        
        if n_of_neighbors in num_of_neighbors:

            num_of_neighbors[n_of_neighbors]+= 1
        else:
            num_of_neighbors[n_of_neighbors] = 1


    print(num_of_neighbors)

join_information_accross_datasets()