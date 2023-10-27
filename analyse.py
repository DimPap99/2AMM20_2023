from Graph import Graph, Subgraph, Node
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
        results.append([Timestep, subgraph.subgraph_id, subgraph.quality, c_il, c_l, c_u, len(nodes)])

        res_str += f"\n\nFull Subgraph: {subgraph}\n"
        res_str += f"size: {len(nodes)} , quality: {subgraph.quality}"
        res_str += f"Illicit: {c_il}, Licit: {c_l} and Unknown: {c_u}\n\n"
        if verbose:
           
            print(f"For subgraph of size: {len(nodes)} , quality: {subgraph.quality}")
            print(f"Illicit: {c_il}, Licit: {c_l} and Unknown: {c_u}")

    if save_results_path is not None:
        with open(save_results_path, "w") as f:
            f.write(res_str)
    return results

def subgraph_list(graph:Graph, subgraph_list):
    result_feat_dict= {}
    subg_result_dict = {}
    for subg in subgraph_list:
        s_id = subg.subgraph_id
        subg_result_dict[s_id] = {}
        for txId in subg.closed:
            node:Node = graph.nodes[txId]
            for key, value in node.features.items():
                if key not in subg_result_dict[s_id]:
                    subg_result_dict[s_id][key] = {}

                if key not in result_feat_dict:
                    result_feat_dict[key] = {}
                if value not in subg_result_dict[s_id][key]:
                     subg_result_dict[s_id][key][value] = 1
                else:
                    subg_result_dict[s_id][key][value] +=1

                if value not in result_feat_dict[key]:
                    result_feat_dict[key][value] = 1
                else:
                    result_feat_dict[key][value] +=1
    return result_feat_dict, subg_result_dict
