import pandas as pd
import csv
import pickle
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
#import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import numpy as np
from math import floor, log2
import matplotlib.pyplot as plot
from Graph import Graph
import networkx as nx
from causalnex.discretiser.discretiser_strategy import MDLPSupervisedDiscretiserMethod
import collections
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


def get_class_pairs(graph, visited_nodes):
    class_pairs = {}
    cls_p = []
    for n in visited_nodes:
        
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



# Create models from data
def best_fit_distribution(data, bins=200, ax=None, verbose=False):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):
        if verbose:
            print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        
        except Exception:
            pass

    
    return sorted(best_distributions, key=lambda x:x[2])

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# Load data from statsmodels datasets
def find_best_fitted_dist_for_features(feature_df:pd.DataFrame, _bins):

    distributions_dict = {}
    size = feature_df.shape[0]
    for feature_col in list(feature_df.columns[5:6]):
        data = feature_df[feature_col]
        best_dist = best_fit_distribution(data, bins=100)
        b = best_dist[0]
        print(b)
        pdf = make_pdf(b[0], b[1], size=size)
        distributions_dict[feature_col] = [b, pdf]
    return distributions_dict




def quality_score(dataframe:pd.DataFrame, subgraph:set, weights_df:pd.DataFrame):
    pass

def create_nx_graph(df_txs_features, df_txs_edgelist):
    all_ids = df_txs_features['txId']

    short_edges = df_txs_edgelist[df_txs_edgelist['txId1'].isin(all_ids)]
    graph = nx.from_pandas_edgelist(short_edges, source = 'txId1', target = 'txId2', 
                                    create_using = nx.DiGraph())
    
    pos = nx.spring_layout(graph)
    df_txs_features['colors'] = df_txs_features['class'].apply(lambda x: "gray" if x==1 else ("Red" if x==2 else "green"))
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)


    node_x = []
    node_y = []
    node_text=[]
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    return graph



def P(feature, feature_categories_count):
    pass



def discretize_per_column(df:pd.DataFrame):
    discretiser = MDLPSupervisedDiscretiserMethod(
        {"min_depth": 5, "random_state": 2020, "min_split": 0, "dtype": int}
        )

    freq_per_feat = {}
    for col in df.columns:
        if col != 'class' and col != 'Time step' and 'txId' not in col:
            print(col)
            discretiser.fit(
            feat_names=[col],
            dataframe=df[[col, "class"]],
            target="class",
            target_continuous=False,
            )
            data = discretiser.transform(df[[col]])
            freq_per_feat[col] =  dict(collections.Counter(data.values.ravel()))
            print(freq_per_feat[col])

    pickle_data('frequencies.pkl', freq_per_feat)
    return freq_per_feat