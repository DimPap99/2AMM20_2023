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




def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def main():
    s = pd.read_csv('A1-dm.csv')
    print("******************************************************")
    print("Entropy Discretization                         STARTED")
    s = entropy_discretization(s)
    print("Entropy Discretization                         COMPLETED")

# This method discretizes attribute A1
# If the information gain is 0, i.e the number of 
# distinct class is 1 or
# If min f/ max f < 0.5 and the number of distinct values is floor(n/2)
# Then that partition stops splitting.
# This method discretizes s A1
# If the information gain is 0, i.e the number of 
# distinct class is 1 or
# If min f/ max f < 0.5 and the number of distinct values is floor(n/2)
# Then that partition stops splitting.
def entropy_discretization(s):

    I = {}
    i = 0
    n = s.nunique()['Class']
    s1 = pd.DataFrame()
    s2 = pd.DataFrame()
    distinct_values = s['Class'].value_counts().index
    information_gain_indicies = []
    print(f'The unique values for dataset s["Class"] are {distinct_values}')
    for i in distinct_values:

        # Step 1: pick a threshold
        threshold = i
        print(f'Using threshold {threshold}')

        # Step 2: Partititon the data set into two parttitions
        s1 = s[s['Class'] < threshold]
        print("s1 after spitting")
        print(s1)
        print("******************")
        s2 = s[s['Class'] >= threshold]
        print("s2 after spitting")
        print(s2)
        print("******************")

        print("******************")
        print("calculating maxf")
        print(f" maxf {maxf(s['Class'])}")
        print("******************")

        print("******************")
        print("calculating minf")
        print(f" maxf {minf(s['Class'])}")
        print("******************")

        print(f"Checking condition a if {s1.nunique()['Class']} == {1}")
        if (s1.nunique()['Class'] == 1):
            break

        print(f"Checking condition b  {maxf(s1['Class'])}/{minf(s1['Class'])} < {0.5} {s1.nunique()['Class']} == {floor(n/2)}")
        if (maxf(s1['Class'])/minf(s1['Class']) < 0.5) and (s1.nunique()['Class'] == floor(n/2)):
            print(f"Condition b is met{maxf(s1['Class'])}/{minf(s1['Class'])} < {0.5} {s1.nunique()['Class']} == {floor(n/2)}")
            break

        # Step 3: calculate the information gain.
        informationGain = information_gain(s1,s2,s)
        I.update({f'informationGain_{i}':informationGain,f'threshold_{i}': threshold})
        print(f'added informationGain_{i}: {informationGain}, threshold_{i}: {threshold}')
        information_gain_indicies.append(i)

    # Step 5: calculate the min information gain
    n = int(((len(I)/2)-1))
    print("Calculating maximum threshold")
    print("*****************************")
    maxInformationGain = 0
    maxThreshold       = 0 
    for i in information_gain_indicies:
        if(I[f'informationGain_{i}'] > maxInformationGain):
            maxInformationGain = I[f'informationGain_{i}']
            maxThreshold       = I[f'threshold_{i}']

    print(f'maxThreshold: {maxThreshold}, maxInformationGain: {maxInformationGain}')

    partitions = [s1,s2]
    s = pd.concat(partitions)

    # Step 6: keep the partitions of S based on the value of threshold_i
    return s #maxPartition(maxInformationGain,maxThreshold,s,s1,s2)


def maxf(s):
    return s.max()

def minf(s):
    return s.min()

def uniqueValue(s):
    # are records in s the same? return true
    if s.nunique()['Class'] == 1:
        return False
    # otherwise false 
    else:
        return True

def maxPartition(maxInformationGain,maxThreshold,s,s1,s2):
    print(f'informationGain: {maxInformationGain}, threshold: {maxThreshold}')
    merged_partitions =  pd.merge(s1,s2)
    merged_partitions =  pd.merge(merged_partitions,s)
    print("Best Partition")
    print("***************")
    print(merged_partitions)
    print("***************")
    return merged_partitions




def information_gain(s1, s2, s):
    # calculate cardinality for s1
    cardinalityS1 = len(pd.Index(s1['Class']).value_counts())
    print(f'The Cardinality of s1 is: {cardinalityS1}')
    # calculate cardinality for s2
    cardinalityS2 = len(pd.Index(s2['Class']).value_counts())
    print(f'The Cardinality of s2 is: {cardinalityS2}')
    # calculate cardinality of s
    cardinalityS = len(pd.Index(s['Class']).value_counts())
    print(f'The Cardinality of s is: {cardinalityS}')
    # calculate informationGain
    informationGain = (cardinalityS1/cardinalityS) * entropy(s1) + (cardinalityS2/cardinalityS) * entropy(s2)
    print(f'The total informationGain is: {informationGain}')
    return informationGain



def entropy(s):
    print("calculating the entropy for s")
    print("*****************************")
    print(s)
    print("*****************************")

    # initialize ent
    ent = 0

    # calculate the number of classes in s
    numberOfClasses = s['Class'].nunique()
    print(f'Number of classes for dataset: {numberOfClasses}')
    value_counts = s['Class'].value_counts()
    p = []
    for i in range(0,numberOfClasses):
        n = s['Class'].count()
        # calculate the frequency of class_i in S1
        print(f'p{i} {value_counts.iloc[i]}/{n}')
        f = value_counts.iloc[i]
        pi = f/n
        p.append(pi)
    
    print(p)

    for pi in p:
        ent += -pi*log2(pi)

    return ent