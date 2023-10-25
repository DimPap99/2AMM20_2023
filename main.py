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
from Beamsearch import beam_search

THRESHOLD = 0.25
TIMESTEPS =[13]#, 42, 35, 32, 29, 22, 9 ,20]

DATASET_BASE = "dataset"
PICKLED_TIMESTEPS = os.path.join("exports", "split_on_timestep.pkl")

df:pd.DataFrame = load_pickled_data("discretized_df.pkl")

main_graph:Graph = load_pickled_data("working_graph.pkl")

initial_beamsearch_candidates = main_graph.get_step_initial_nodes()
print(len(initial_beamsearch_candidates))
beam_candidates = []
for candidate in initial_beamsearch_candidates:
    s = Subgraph()
    s.closed.add(candidate)
    
    s.quality = s.calculate_quality(main_graph, THRESHOLD)
    if s.quality > 0:
        beam_candidates.append(s)

print(len(beam_candidates))

beam_search(main_graph, 4, beam_candidates)



