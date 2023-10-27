import numpy as np
import math
from Graph import Graph, Node, Subgraph, dataclasses
import pandas as pd


def create_initial_candidates(graph:Graph, threshold):
    initial_beamsearch_candidates = graph.get_step_initial_nodes()
    subgraph_candidates = []

    for candidate in initial_beamsearch_candidates:
        s = Subgraph()
        s.closed.add(candidate)
        s.quality = s.calculate_quality(graph, threshold)
        if s.quality > 0:
            candidate_children:set = graph.nodes[candidate].children
            s.open = s.open.union(candidate_children)
            subgraph_candidates.append(s)
    return subgraph_candidates


def mine_graphs(graph:Graph, threshold, beam_edge_width=3, beam_search_subgraphs=False, top_k=None, verbose=False, min_size=None, mine_unexplored=False, max_size=None):
    
    #run the first iteration
    subgraph_candidates =  create_initial_candidates(graph, threshold)
    interesting_graphs, rejected_candidates = beam_search_edges(graph=graph, threshold=threshold, beam_width=beam_edge_width, initial_candidates=subgraph_candidates, verbose=verbose, mine_unexplored=mine_unexplored)
    subgraph_candidates = []

    while len(rejected_candidates) != 0:
        for candidate in rejected_candidates:
            s = Subgraph()
            s.closed.add(candidate)
            s.quality = s.calculate_quality(graph, threshold)
            if s.quality > 0:
                candidate_children:set = graph.nodes[candidate].children
                s.open = s.open.union(candidate_children)
                subgraph_candidates.append(s)

        subgraph_results, new_rejected_candidates = beam_search_edges(graph=graph, threshold=threshold, beam_width=4, initial_candidates=subgraph_candidates, verbose=verbose, mine_unexplored=True)
        interesting_graphs = interesting_graphs + subgraph_results
        rejected_candidates = new_rejected_candidates.difference(rejected_candidates)
        subgraph_candidates = []
    

    unique_interesting_subgraphs = []
    unique_sub_graphs = set()
    #filter graphs that are duplicate and also based on size
    for subgraph in interesting_graphs:
        if min_size is not None:
            if len(subgraph.closed) < min_size:
                continue
        if max_size is not None:
            if len(subgraph.closed) > max_size:
                continue
        subgraph.compute_subgraph_unique_id()
        if subgraph.subgraph_id not in unique_sub_graphs:
            
            unique_sub_graphs.add(subgraph.subgraph_id)
            #count total illicit, licit and unknown nodes
            subgraph.calculate_category_totals(graph)
            unique_interesting_subgraphs.append(subgraph)

    #Not implemented yet
    if top_k is not None and top_k > 0:
        if top_k <= len(unique_interesting_subgraphs):
            return unique_interesting_subgraphs[0:top_k]
    if verbose:
        print(f"Generated in total: {len(unique_interesting_subgraphs)}")
    
    return unique_interesting_subgraphs

def beam_search_edges(graph:Graph, threshold, beam_width=3, initial_candidates:Subgraph=None, verbose=False, mine_unexplored=False):
    """
    Perform beam search for graph mining to find frequent subgraphs.

    Parameters:
    - graph: The input graph represented as an adjacency matrix.
    - max_pattern_size: Maximum size of subgraphs to search for.
    - beam_width: Width of the beam (number of candidates to keep at each step).


    Returns:
    - A list of frequent subgraphs.
    """

    print(f"Will start expanding {len(initial_candidates)} graphs with a beam width of {beam_width}")
    # Initialize the beam with single nodes as candidates
    subgraph_candidates = initial_candidates

    interesting_subgraphs = []
    #while CAN_EXPAND_MORE:
    rejected_candidates = set()
    # Iterate over the pattern size

    iterations = 0
    counts_per_iteration = 0
    while len(subgraph_candidates) > 0:
        # Generate candidates for each current subgraph in the beam
        for i in range(0, len(subgraph_candidates) ):
            subgraph:Subgraph = subgraph_candidates[i]
            
            # Generate candidate subgraphs
            generated_candidates, unexplored_candidates = generate_candidate_subgraph(subgraph=subgraph, graph=graph, threshold=threshold, beam_width=beam_width)
            
            if len(generated_candidates) == 0:
                counts_per_iteration += 1
                subgraph_candidates[i].can_be_expanded = False
                if verbose:
                    print(f"Mined subgraph of length: {len(subgraph.closed)} and quality: {subgraph.quality}")
            for candidate in generated_candidates:
                # merge the candidate subgraphs with the original subgraph until they are all merge or until
                # the quality score reduces
                
                prev_quality = subgraph.quality
                # if(len(subgraph.closed) > 100):
                #     print("t")
                subgraph.merge(candidate, graph, threshold)
                if subgraph.quality > prev_quality:
                    subgraph_candidates[i] = subgraph
                else:
                    rejected_candidates.add(candidate.closed[-1])
                        
            if mine_unexplored:
                rejected_candidates = rejected_candidates.union(unexplored_candidates)
        
            
        interesting_subgraphs = interesting_subgraphs + [x for x in subgraph_candidates if x.can_be_expanded == False]
        subgraph_candidates = [x for x in subgraph_candidates if x.can_be_expanded == True]
        if verbose:
            print(f"Mined {counts_per_iteration} graphs in iteration number {iterations}")
        counts_per_iteration = 0
        iterations += 1

    return interesting_subgraphs, rejected_candidates

def generate_candidate_subgraph(subgraph:Subgraph, graph:Graph, threshold:float, beam_width):
    """
    Generate candidate subgraphs from the current subgraph.

    Parameters:
    - subgraph: The current subgraph.
    - graph: The input graph represented as an adjacency matrix.

    Returns:
    - A list of candidate subgraphs.
    """
    candidate_subgraphs = []
    current_quality = subgraph.calculate_quality(graph=graph, threshold=threshold)
    unexplored_candidates = set()
    
    # Implement this function to generate candidate subgraphs.
    for candidate in subgraph.open.copy():
        #add one of the neighbors of the original subgraph into a new temp subgraph
        #and calculate quality

        temporary_subgraph:Subgraph = dataclasses.replace(subgraph)
        temporary_subgraph.closed.add(candidate)
        temp_subgraph_score = temporary_subgraph.calculate_quality(graph=graph, threshold=threshold)
        temporary_subgraph.open.remove(candidate)
        #if the new subgraph has better quality than the original one add it to the candidates
        if temp_subgraph_score >= current_quality:
            candidate_children:set = graph.nodes[candidate].children
            for child in candidate_children:
                if child not in temporary_subgraph.closed:
                    #the children of the candidate to the open set
                    temporary_subgraph.open = temporary_subgraph.open.union(candidate_children)
            candidate_subgraphs.append(temporary_subgraph)
        else:
            unexplored_candidates.add(candidate)
            del temporary_subgraph
            continue
    #sort the candidate subgraphs based on the subgraph quality
    #and keep the k best subgraphs (based on beam width)
    if len(candidate_subgraphs) > 0:
        candidate_subgraphs.sort(key=lambda x: x.quality, reverse=True)
        if len(candidate_subgraphs) >= beam_width:
            return candidate_subgraphs[0:beam_width], unexplored_candidates
        
    return candidate_subgraphs, unexplored_candidates
        


def merge_subgraphs(graph:Graph, subgraph:Subgraph, candidate_subgraph:Subgraph, threshold):
    """
    Merges an original subgraphs with subgraphs that
    have been generated from it and raise the total quality measure.

    """
    subgraph.open = subgraph.open.union(candidate_subgraph.open)
    subgraph.closed = subgraph.closed.union(candidate_subgraph.closed)
    subgraph.calculate_quality(graph, threshold)
    return subgraph

