from Graph import Graph, Subgraph
import os

def get_num_of_nodes_per_cat(Timestep, graph:Graph, subgraphs_list:list, verbose=False, save_results_path=None):
    results = []
    res_str= ""
    for subgraph in subgraphs_list:
        nodes = subgraph.closed
        c_il, c_l, c_u = 0, 0 ,0
        for n in nodes:
            category = graph.nodes[n].label
            if category == 1:
                c_il += 1
            elif category == 2:
                c_l += 1
            else:
                c_u += 1
        results.append([Timestep, subgraph.subgraph_id, subgraph.quality, subgraph.quality_without_size_penalty, c_il, c_l, c_u])

        res_str += f"\n\nFull Subgraph: {subgraph}\n"
        res_str += f"size: {len(nodes)} , quality: {subgraph.quality} and score without penalty: {subgraph.quality_without_size_penalty}\n"
        res_str += f"Illicit: {c_il}, Licit: {c_l} and Unknown: {c_u}\n\n"
        if verbose:
           
            print(f"For subgraph of size: {len(nodes)} , quality: {subgraph.quality} and score without penalty: {subgraph.quality_without_size_penalty}")
            print(f"Illicit: {c_il}, Licit: {c_l} and Unknown: {c_u}")

    if save_results_path is not None:
        with open(save_results_path, "w") as f:
            f.write(res_str)
    return results