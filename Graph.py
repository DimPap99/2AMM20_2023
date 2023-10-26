import math
BASE = 2
from ordered_set import OrderedSet
import sys
class Node(object):
    """
        The node class represents a single node (transaction) in the transactions graph
        :attribute features: Contains the anonymised features provided by elliptic(List)
        :attribute children: outgoing transactions from this node  (List of txIds - ints)
        :attribute txId: Transaction Id (string);
        :attribute timestep: The timestep in which this transaction belongs (int);
        :attribute parents: The previous (parent) nodes in the chain of nodes. (List of txIds - ints)
        :attribute label: The label of the transaction Illicit|licit|Uknown (string);
    """

    def __init__(self, feats_dict, id, ts, l) -> None:
        self.features:dict = feats_dict
        self.children = set()
        self.parents = set()
        self.txId = id #the first element of features is always the txId
        self.timestep = ts #the second element in always the timestep.
        self.label = l 
    
    
    def get_num_of_neighbors(self, only_children=False):
        """
        Return the neighbors of this node
        """
        if only_children:
            return len(self.children)
        else:
            return len(self.children) + len(self.parents)
    
    def total_neighbors(self):
        """
        Returns all the nodes associated with this node, parents and children
        """
        return self.parents + self.children
    
    
    def get_edge_pairs(self):
        """
            Gets all the pair of edges between the current node and its neighbors
        """
        edge_pairs = set()

        if len(self.children) != 0:
            for childTxId in self.children:
                edge_pairs.add((self.txId, childTxId))
        
        return edge_pairs
#A nodes tx.id is the key and the value is a node object
class Graph(object):
    """
        A representation of the acyclic transaction graph.
        :attribute Nodes: Contains all the nodes. (Dict ( txId -> Node obj))
    """
    feature_importances = {}
    feature_occurences = {}
    def __init__(self) -> None:
        self.nodes = {}
        self.edges_from = {}
        self.edges_to = {}
        self.total_edges = set()
        self.node_txId_set = set()
    def add_Node(self, node:Node):
        if node.txId not in self.nodes:
            self.nodes[node.txId] = node
            self.node_txId_set.add(node.txId)
    def get_step_initial_nodes(self):
        candidates = set()
        for key, value in self.nodes.items():
            node:Node = value
            #no Parrent and at least 1 child
            if len(node.parents) == 0 and len(node.children) >= 1:
                candidates.add(key)
        return candidates
    
    def get_total_nodes(self):
        return len(self.nodes.keys())

    def create_from_lists(self, timestep_cols, features, edge_list):
        """
            Creates a graph object from a list of features and a list of edges.
            :attribute features: A list with the node features. A merge with a pandas dataframe has first occured so that it also includes the class in the list's last index.
            :attribute features: A list with node edges.

        """
        for row in features:
            txId = int(row[0])
            ts = int(row[2])
            l = row[-1]
            if txId not in self.nodes:
                
                feat_dct = {}
                i = 0
                for col in timestep_cols:
                    feat_dct[col] = row[i]
                    i +=1
                feat_dct.pop('txId')
                feat_dct.pop('class')
                feat_dct.pop('Time step')
                self.add_Node(Node(feat_dct, txId, ts, l))

        for edge in edge_list:
            curr_txId = int(edge[0])
            neighbor = int(edge[1])
            if neighbor in self.nodes and curr_txId in self.nodes:
                self.add_edge(curr_txId, neighbor)

    def add_edge(self, current_txId, neighbor_txId):
        """
            Adds a node in the graphs and creates the associations between its neighbors
            :attribute current_txId: The txId of the current node (int).
            :attribute neighbor_txId: The txId of the neighbor_txId node (int).

        """
        self.total_edges.add((current_txId, neighbor_txId))
        self.nodes[current_txId].children.add(neighbor_txId) #add the neighbor to the current node neighbor set
        self.nodes[neighbor_txId].parents.add(current_txId) #add the current node as the node that preceds the neighbor node
        
    def graph_info(self, graph_number=None, return_output=True, isSuperSet=False):
        c_label1, c_label2, c_label3 = 0, 0, 0#the amount of illicit, licit and unknown labels in a graph
        node_number_of_neighbors = {} #the number of nodes that have x amount of neightbors (key -> number of neighbors : value -> amount of nodes)
        node_number_of_children_neighbors = {} #the number of nodes that have x amount of neightbors (key -> number of neighbors : value -> amount of nodes)
        print_str = ""
        for k, v in self.nodes.items():
            node:Node = v
            if node.label == 1:
                c_label1 += 1
            elif node.label == 2:
                c_label2 += 1
            else:
                c_label3 +=1
            

            n_of_neighbors = node.get_num_of_neighbors()
            n_of_children_neighbors = node.get_num_of_neighbors(only_children=True)
            #find number of children and parent neighbors
            if n_of_neighbors in node_number_of_neighbors:

                node_number_of_neighbors[n_of_neighbors]+= 1
            else:
                node_number_of_neighbors[n_of_neighbors] = 1

            if n_of_children_neighbors in node_number_of_children_neighbors:

                node_number_of_children_neighbors[n_of_children_neighbors]+= 1
            else:
                node_number_of_children_neighbors[n_of_children_neighbors] = 1

        print_str +=f"\n\n######################################## For{f' superset' if isSuperSet else ''} graph {'on timestemp ' + str(graph_number) if graph_number is not None else ''} ########################################\n" 
        print_str += f"\nNumber of labels per category:\nLabel 1: {c_label1} - Label 2: {c_label2} - Label 3: {c_label3}\n"
        print_str +="\nNumber of nodes with x amount of neighbors (Parents and Children): \n"
        print_str +=str(node_number_of_neighbors)
        print_str +="\n\nNumber of nodes with x amount of neighbors (Only Children): \n"
        print_str +=str(node_number_of_children_neighbors)
        print(print_str)
        if return_output:
            return print_str

def P(g:Graph, feature, feature_value):
    try:
        counts:dict = g.feature_occurences[feature]
        total_category_occurences = counts[feature_value]
        total_samples = sum(g.feature_occurences[feature].values())
        return total_category_occurences/total_samples

    except Exception as e:
        print(f"Couldnt find feature: {feature} and value {feature_value}")
        sys.exit(0)
    

def TD(threshold, g:Graph, feature, feature_value):
    poss =  P(g, feature, feature_value)
    return threshold - poss
import dataclasses

@dataclasses.dataclass                    
class Subgraph:
    open: set = dataclasses.field(default_factory=set) 
    closed: OrderedSet = dataclasses.field(default_factory=OrderedSet) 
    quality: float = 0
    can_be_expanded: bool = True        

    def calculate_quality(self, graph:Graph, threshold) -> float:
        score = 0
        for key, value in graph.feature_importances.items():
            for txId in self.closed:
                current_node_feature_value = graph.nodes[txId].features[key]
                td_val = TD(threshold, graph, key, current_node_feature_value)
                score += td_val * (1 + value) 
        self.quality = score 
        if len(self.closed) > 1:
            self.quality = self.quality / (math.log(len(self.closed), BASE)**2)
        return self.quality
    

    def merge(self, subgraph, graph:Graph, threshold):
        self.open = self.open.union(subgraph.open)
        self.closed = self.closed.union(subgraph.closed)
        self.calculate_quality(graph, threshold)
        
