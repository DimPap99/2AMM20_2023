from helpers import read_csv, export_csv
from Graph import Graph, Node
import os

DATASET_BASE = "dataset"
txs_classes = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_classes.csv"))
txs_edgelist = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_edgelist.csv"))
txs_features =  read_csv(os.path.join(DATASET_BASE, "elliptic_txs_features.csv"))

#Prepare two datasets. 1 with all the connected nodes. 1 with all the connected nodes that belong to either class 1 or 2
def join_information_accross_datasets():
    graph = Graph()
    #initialize nodes with features
    for row in txs_features:
        graph.add_Node(Node(features=row))
    
    #assign class to nodes
    for row in txs_classes:
        txId = row[0]
        label = 3 if row[2] == "Uknown" else row[2]
        if txId in graph.nodes:
            graph.nodes[txId].label = label
    
    #find neighboring nodes
    for row in txs_edgelist:
        current_txId = row[0]
        neighboring_txId = row[2]
        if current_txId in graph.nodes and neighboring_txId in graph.nodes:
            graph.nodes[current_txId].add_neighbor(current_txId, neighboring_txId)