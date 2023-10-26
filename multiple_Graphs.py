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
import sys
THRESHOLD = 0.25
TIMESTEPS =[13, 42, 35, 32, 29, 22, 9 ,20]

DATASET_BASE = "dataset"
PICKLED_TIMESTEPS = os.path.join("exports", "split_on_timestep.pkl")


graph_dict:dict = load_pickled_data("working_graphs_dict.pkl")
main_graph:Graph = graph_dict[13]


initial_beamsearch_candidates = main_graph.get_step_initial_nodes()

beam_candidates = []

for candidate in initial_beamsearch_candidates:
    s = Subgraph()
    s.closed.add(candidate)
    s.quality = s.calculate_quality(main_graph, THRESHOLD)
    if s.quality > 0:
        candidate_children:set = main_graph.nodes[candidate].children
        s.open = s.open.union(candidate_children)
        beam_candidates.append(s)
interesting_graphs, rejected_candidates = beam_search(graph=main_graph, threshold=THRESHOLD, beam_width=4, initial_candidates=beam_candidates)
beam_candidates = []

while len(rejected_candidates) != 0:
    for candidate in rejected_candidates:
        s = Subgraph()
        s.closed.add(candidate)
        s.quality = s.calculate_quality(main_graph, THRESHOLD)
        if s.quality > 0:
            candidate_children:set = main_graph.nodes[candidate].children
            s.open = s.open.union(candidate_children)
            beam_candidates.append(s)

    subgraph_results, new_rejected_candidates = beam_search(graph=main_graph, threshold=THRESHOLD, beam_width=4, initial_candidates=beam_candidates)
    interesting_graphs = interesting_graphs + subgraph_results
    rejected_candidates = new_rejected_candidates.difference(rejected_candidates)
    beam_candidates = []

sz_dict = {}
for subgraph in interesting_graphs:
    l = len(subgraph.closed)
    if l not in sz_dict:
        sz_dict[l] = 1
    else:
        sz_dict[l]+=1
print(sz_dict)