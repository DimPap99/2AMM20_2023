import pandas as pd
import csv
import pickle

def pickle_data(file_path, obj, verbose=False):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        if verbose:
            print(f"Succesfully pickled the object on file {file_path}")
    except Exception as e:
        print(f"An exception occured while pickling an object:\n{e}")

def load_pickled_data(file_path, verbose=False):
    try:
        data = None
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if verbose:
            print(f"Succesfully loaded a pickled object from file {file_path}")
        return data
    except Exception as e:
        print(f"An exception occured while loading a pickled object:\n{e}")

def read_csv(file_path:str, ret_Dataframe=False, verbose=False, drop_header=False):
    try:
        if verbose:
            print(f"Reading the csv file: {file_path}")
        if ret_Dataframe:
            return pd.read_csv(file_path)
        else:
            data = []
            with open(file_path, 'r') as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    data.append(row)
        print("Finished reading csv file...")
        if drop_header is True:
            return data[1:]
        else:
            return data
    except Exception as err:
        print(f"An error occured while attempting to read: {file_path}")
        print(err)



def export_csv(file_path:str, data, isDataframe = False, headers=None, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, encoding="utf-8", verbose=False):
    try:
        if verbose:
            print(f"Will write to the csv file: {file_path}")
        if isDataframe:
            data.to_csv(file_path, sep=delimiter, encoding=encoding)
        else:

            if headers is not None:
                writer = csv.writer(file_path, fieldnames=headers, delimiter=delimiter, quotechar=quoting)
                writer.writeheader()
            else:
                writer = csv.writer(file_path, fieldnames=headers, delimiter=delimiter, quotechar=quoting)
                for row in data:
                    writer.writerow(row)
        if verbose:
            print(f"Finished writing to {file_path}. Rows written: {len(data)}")
    except Exception as err:
        print(f"An error occured while attempting to write to: {file_path}")
        print(err)


def get_class_pairs(graph, edge_pairs):
    class_pairs = {}
    cls_p = []
    for pair in edge_pairs:
        n = int(pair[1])
        cls_p.append(graph.nodes[n].label)
    return cls_p

from collections import defaultdict, deque

def choose_node_based_on_neighbors(graph, n, only_children=True, label=None):
    num_of_neighbors = n
    node = None
    for key, value in graph.nodes.items():
        txId = key
        node = value
        if label is None:
            if node.get_num_of_neighbors(only_children=only_children) == n:
                break
        else:
            if label == label and node.get_num_of_neighbors(only_children=only_children) >= n:
                break
    return node
    
def bfs(graph, start, keep_direction=False):
            """
                Simple bfs implementation for dataset exploration. Also keeps a set of all node members 
            """
            visited = set()
            queue = deque([start])
            pairs = set()
            
            while queue:
                node_txId = queue.popleft()
                if node_txId not in visited:
                    visited.add(node_txId)
                    neighbors = graph.nodes[node_txId].total_neighbors() if not keep_direction else graph.nodes[node_txId].children
                    for neighbor in neighbors:
                        pairs.add((node_txId, neighbor))
                        if neighbor not in visited:
                            queue.append(neighbor)
            return pairs, visited