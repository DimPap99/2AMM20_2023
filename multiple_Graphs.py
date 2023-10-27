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
from helpers import export_csv, pickle_data

NUM_OF_PROCESSES = 5
graph_dict:dict = load_pickled_data("working_graphs_dict2.pkl")
AVAILABLE_TIMESTEPS = list(graph_dict.keys())
print(AVAILABLE_TIMESTEPS)
THRESHOLD = 0.2
#TIMESTEPS = [9, 13, 15, 16, 17, 20, 21, 22, 24, 25, 26, 29, 31, 32, 35, 38, 40, 41, 42] #---> those timesteps contain most illicit nodes
MIN_SUBGRAPH_SIZE = 12
MAX_SIZE = None
MINE_UNEXPLORED = True
def mine_timesteps_in_parallel(Timestep):

    print(f"Starting job for Timestep {Timestep}...")
    SAVE_RESULTS_DIR = os.path.join("Results", f"Timestep_{Timestep}.txt")

    main_graph:Graph = graph_dict[Timestep]
    interesting_subgraphs = mine_graphs(main_graph, THRESHOLD, 4, verbose=False, min_size=MIN_SUBGRAPH_SIZE, mine_unexplored=MINE_UNEXPLORED, max_size=None)
    print(f"Timestep-{Timestep}. Finished mining: {len(interesting_subgraphs)} subgraphs with min_size: {MIN_SUBGRAPH_SIZE}")
    
   
    res = get_num_of_nodes_per_cat(Timestep, main_graph, interesting_subgraphs, verbose=False, save_results_path=SAVE_RESULTS_DIR)
    
    print(f"Finished job for Timestep {Timestep}...")
    return res, interesting_subgraphs

if __name__ == '__main__':
    with Pool(NUM_OF_PROCESSES) as p:
        
        results = p.map(mine_timesteps_in_parallel, AVAILABLE_TIMESTEPS)

    csv_data = []
    result_subgraphs = []

    _headers =["TimeStep", "Sid", "Quality", "Illicit", "Licit", "Unknown", "size"]

    for result in results:
        for l in result[0]:
            csv_data.extend([l])
        for s in result[1]:
            result_subgraphs.append(s)

    export_csv(file_path=f"Results_sz__min{MIN_SUBGRAPH_SIZE}_max_{MAX_SIZE}_thresh_{str(THRESHOLD).replace('.','_')}_{MINE_UNEXPLORED}.csv",  data=csv_data, isDataframe=False, headers=_headers)
    pickle_data(file_path=f"Subgraphs__min{MIN_SUBGRAPH_SIZE}_max_{MAX_SIZE}_thresh_{str(THRESHOLD).replace('.','_')}_{MINE_UNEXPLORED}.pkl",  obj=result_subgraphs, verbose=True)