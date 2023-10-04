
class Node(object):
    """
        The node class represents a single node (transaction) in the transactions graph
        :attribute features: Contains the anonymised features provided by elliptic(List)
        :attribute outgoing_edges: outgoing transactions from this node  (List of txIds - strings)
        :attribute incoming_edges: incoming transactions from this node  (List of txIds - strings)
        :attribute txId: Transaction Id (string);
        :attribute timestep: The timestep in which this transaction belongs (int);
        :attribute prev_node_txId: The tx_Id of the previous node in the chain of nodes. (string)
        :attribute label: The label of the transaction Illicit|licit|Uknown (string);
    """

    def __init__(self, feats) -> None:
        self.features = feats
        self.outgoing_edges = []
        self.incoming_edges = []
        self.txId = feats[0] #the first element of features is always the txId
        self.timestep = feats[1] #the second element in always the timestep.
        self.prev_nodes_txIds = []
        self.label = None
    
    def get_num_of_neighbors(self):
        return len(self.outgoing_edges) + len(self.incoming_edges)
    def unpack(self):
        """
            Unpacks the object into a list the contains the txid, the corresponding features, class, in one list
        """



#A nodes tx.id is the key and the value is a node object
class Graph(object):
    """
        A representation of the acyclic transaction graph.
        :attribute Nodes: Contains all the nodes. (Dict ( txId -> Node obj))
    """

    def __init__(self) -> None:
        self.nodes = {}

    def add_Node(self, node:Node):
        self.nodes[node.txId] = node

    def add_neighbor_to_node(self, current_txId, neighbor_txId):
        self.nodes[current_txId].outgoing_edges.append(neighbor_txId) #add the neighbor to the current node neighbor list
        self.nodes[neighbor_txId].incoming_edges.append(current_txId) #add the current node as the node that preceds the neighbor node

    def export_graph_nodes(only_labeled=True):
        pass