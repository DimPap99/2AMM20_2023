
class Node(object):
    """
        The node class represents a single node (transaction) in the transactions graph
        :attribute features: Contains the anonymised features provided by elliptic(List)
        :attribute neighbors: neighboring transactions (List of nodes)
        :attribute txId: Transaction Id (string);
        :attribute timestep: The timestep in which this transaction belongs (int);
        :attribute group_no: Internal variable used to indicate in which group of nodes the current node belongs (int);
        :attribute prev_node_txId: The tx_Id of the previous node in the chain of nodes. (string)
        :attribute label: The label of the transaction Illicit|licit|Uknown (string);
    """

    def __init__(self, features) -> None:
        self.features = features
        self.neighbors = []
        self.txId = features[0] #the first element of features is always the txId
        self.timestep = features[1] #the second element in always the timestep.
        self.prev_nodes_txIds = []
        self.group_no = -1
        self.lalbel = None
    def unpack(self):
        """
            Unpacks the object into a list the contains the txid, class and the corresponding features in one list
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
    def add_neighbor(self, current_txId, neighbor_txId):
        self.nodes[current_txId].neighbors.append(neighbor_txId) #add the neighbor to the current node neighbor list
        self.nodes[neighbor_txId].prev_nodes_txIds.append(current_txId) #add the current node as the node that preceds the neighbor node
    def export_graph_nodes(only_labeled=True):
        pass