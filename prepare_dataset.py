from helpers import read_csv, export_csv, pickle_data, load_pickled_data
from Graph import Graph, Node
import os
import pandas as pd
from visualizations import visualize_graph
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
        timestep = int(row[1])
        if timestep not in split_on_timestep:
            split_on_timestep[timestep] = Graph()
        else:
            split_on_timestep[timestep].add_Node(node)
    
    #assign class to nodes
    for row in txs_classes:
        txId = int(row[0])
    
        if txId in superset_graph.nodes:
            superset_graph.nodes[txId].label = int(row[1])
    #find neighboring nodes
    for row in txs_edgelist:
        current_txId = int(row[0])
        neighboring_txId = int(row[1])
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


# txs_classes:pd.DataFrame = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_classes.csv"), ret_Dataframe=True)
# txs_edgelist:pd.DataFrame = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_edgelist.csv"), ret_Dataframe=True)
# txs_features:pd.DataFrame =  read_csv(os.path.join(DATASET_BASE, "elliptic_txs_features.csv"), ret_Dataframe=True)

# txs_by_class = txs_classes.groupby('class').count()

# merge_feats_class = pd.merge(txs_features, txs_classes, on='txId')
# graph_merged = pd.merge(merge_feats_class, txs_edgelist, left_on='txId', right_on='txId1')
# pickle_data("merged_dataframe.pkl", graph_merged, True)

# print(txs_by_class)
#visualize_graph(merge_feats_class, txs_edgelist, 20)

