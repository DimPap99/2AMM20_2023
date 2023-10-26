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
from Graph import Graph, Subgraph
from GraphMining import mine_graphs
import sys
from analyse import get_num_of_nodes_per_cat
from multiprocessing import Pool, Manager
from helpers import export_csv

NUM_OF_PROCESSES = 4
PICKLED_TIMESTEPS = os.path.join("exports", "split_on_timestep.pkl")
graph_dict:dict = load_pickled_data("working_graphs_dict.pkl")
AVAILABLE_TIMESTEPS = list(graph_dict.keys())
THRESHOLD = 0.25
TIMESTEPS = [9, 13, 15, 16, 17, 20, 21, 22, 24, 25, 26, 29, 31, 32, 35, 38, 40, 41, 42]
MIN_SUBGRAPH_SIZE = 20

def mine_timesteps_in_parallel(Timestep):

    print(f"Starting job for Timestep {Timestep}...")
    SAVE_RESULTS_DIR = os.path.join("Results", f"Timestep_{Timestep}.txt")

    main_graph:Graph = graph_dict[Timestep]
    interesting_subgraphs = mine_graphs(main_graph, THRESHOLD, 3, verbose=False, min_size=MIN_SUBGRAPH_SIZE)
    print(f"Timestep-{Timestep}. Finished mining: {len(interesting_subgraphs)} subgraphs with min_size: {MIN_SUBGRAPH_SIZE}")
    
   
    res = get_num_of_nodes_per_cat(Timestep, main_graph, interesting_subgraphs, verbose=False, save_results_path=SAVE_RESULTS_DIR)
    
    print(f"Finished job for Timestep {Timestep}...")
    return res

if __name__ == '__main__':
    with Pool(NUM_OF_PROCESSES) as p:
        
        results = p.map(mine_timesteps_in_parallel, AVAILABLE_TIMESTEPS)
    csv_data = []
    for l in results:
        csv_data.extend(l)

    export_csv(file_path=f"Results_sz_{MIN_SUBGRAPH_SIZE}_thresh_{str(THRESHOLD).replace('.','_')}.csv",  data=csv_data, isDataframe=False, headers=["TimeStep", "Sid", "Quality", "QualityNoPenalty", "Illicit", "Licit", "Unknown"])