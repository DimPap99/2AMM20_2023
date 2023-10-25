import numpy as np
import math
from Graph import Graph, Node, Subgraph, dataclasses

def beam_search(graph:Graph, threshold, beam_width=3, initial_candidates:Subgraph=None):
    """
    Perform beam search for graph mining to find frequent subgraphs.

    Parameters:
    - graph: The input graph represented as an adjacency matrix.
    - max_pattern_size: Maximum size of subgraphs to search for.
    - beam_width: Width of the beam (number of candidates to keep at each step).


    Returns:
    - A list of frequent subgraphs.
    """

    CAN_EXPAND_MORE = True
    # Initialize the beam with single nodes as candidates
    beam = initial_candidates

    interesting_subgraphs = []
    #while CAN_EXPAND_MORE:
        
    # Iterate over the pattern size
    new_beam = []
    subgraph_can_be_expanded = [True] * len(beam)
    # Generate candidates for each current subgraph in the beam
    for i in range(0, len(beam) -1 ):
        if subgraph_can_be_expanded[i]:
            subgraph:Subgraph = beam[i]
            # Generate candidate subgraphs
            candidates = generate_candidate_subgraph(subgraph, graph)
            if len(candidates) == 0:
                subgraph_can_be_expanded[i] = False
            for candidate in candidates:
                # merge the candidate subgraphs with the original subgraph until they are all merge or until
                # the quality score reduces
                
                prev_quality = subgraph.quality
                subgraph.merge(candidate, graph, threshold)
                if subgraph.quality >= prev_quality:
                    beam[i] = subgraph
                else:
                    break
                    

            # Select the top candidates based on support
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]

            # Store frequent subgraphs of the current size
            interesting_subgraphs.extend([(subgraph, support) for subgraph, support in beam])
    
    return interesting_subgraphs

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

    
    # Implement this function to generate candidate subgraphs.
    for candidate in subgraph.open:
        #add one of the neighbors of the original subgraph into a new temp subgraph
        #and calculate quality

        temporary_subgraph:Subgraph = dataclasses.replace(subgraph)
        temporary_subgraph.closed.add(candidate)
        temp_subgraph_score = temporary_subgraph.calculate_quality(graph=graph, threshold=threshold)
        #if the new subgraph has better quality than the original one add it to the candidates
        if temp_subgraph_score >= current_quality:
            temporary_subgraph.open.remove(candidate)
            candidate_children:set = graph.nodes[candidate].children
            #the children of the candidate to the open set
            temporary_subgraph.open.union(candidate_children)
            candidate_subgraphs.append(temporary_subgraph)
        else:
            del temporary_subgraph
            continue
    #sort the candidate subgraphs based on the subgraph quality
    #and keep the k best subgraphs (based on beam width)
    if len(candidate_subgraphs) > 0:
        candidate_subgraphs.sort(key=lambda x: x.quality, reverse=True)
        if len(candidate_subgraphs) >= beam_width:
            return candidate_subgraphs[0:beam_width]
        
    return candidate_subgraphs
        


def merge_subgraphs(graph:Graph, subgraph:Subgraph, candidate_subgraph:Subgraph, threshold):
    """
    Merges an original subgraphs with subgraphs that
    have been generated from it and raise the total quality measure.

    """
    subgraph.open.union(candidate_subgraph.open)
    subgraph.closed.union(candidate_subgraph.closed)
    subgraph.calculate_quality(graph, threshold)
    return subgraph

