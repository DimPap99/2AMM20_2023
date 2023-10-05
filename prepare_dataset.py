from helpers import read_csv, export_csv, pickle_data, load_pickled_data
from Graph import Graph, Node
import os

DATASET_BASE = "dataset"

#Prepare two datasets. 1 with all the connected nodes. 1 with all the connected nodes that belong to either class 1 or 2
def join_information_accross_datasets(keep_graphs_by_step=False):
    txs_classes = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_classes.csv"), drop_header=True)
    txs_edgelist = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_edgelist.csv"), drop_header=True)
    txs_features =  read_csv(os.path.join(DATASET_BASE, "elliptic_txs_features.csv"), drop_header=True)
   
    split_on_timestep = {}
    superset_graph = Graph()
    #initialize nodes with features
    for row in txs_features:
        node = Node(feats=row)
        superset_graph.add_Node(node)
        timestep = row[1]
        if timestep not in split_on_timestep:
            split_on_timestep[timestep] = Graph()
        else:
            split_on_timestep[timestep].add_Node(node)
    
    #assign class to nodes
    for row in txs_classes:
        txId = row[0]
    
        if txId in superset_graph.nodes:
            superset_graph.nodes[txId].label = int(row[1])
    #find neighboring nodes
    for row in txs_edgelist:
        current_txId = row[0]
        neighboring_txId = row[1]
        if current_txId in superset_graph.nodes and neighboring_txId in superset_graph.nodes:
            superset_graph.add_edge(current_txId, neighboring_txId)
    pickle_data("superset_graph.pkl", superset_graph, True)
    pickle_data("split_on_timestep.pkl", split_on_timestep, True)

    analyse_graphs(superset_graph, split_on_timestep, export_to_txt=True)
    
    
   
def analyse_graphs(superset_graph, split_on_timestep=None, export_to_txt=False):
    superset_str = superset_graph.graph_info(isSuperSet=True, return_output=True)
    
    export_timesteps_txt = ""

    if split_on_timestep is not None:
        num_of_graph = 1
        for k, v in split_on_timestep.items():
            export_timesteps_txt += v.graph_info(graph_number=num_of_graph, return_output=True)
            num_of_graph+=1

        if export_timesteps_txt:
            with open("timestep_graphs.txt", 'w') as f:
                f.write(export_timesteps_txt)
                
    if export_timesteps_txt:
        with open("supertset_graph.txt", 'w') as f:
            f.write(superset_str)
        
join_information_accross_datasets()
