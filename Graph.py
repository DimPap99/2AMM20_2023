
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
        self.children = []
        self.parents = []
        self.txId = feats[0] #the first element of features is always the txId
        self.timestep = feats[1] #the second element in always the timestep.
        self.prev_nodes_txIds = []
        self.label = None
    
    def get_num_of_neighbors(self, only_children=False):
        if only_children:
            return len(self.children)
        else:
            return len(self.children) + len(self.parents)
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
        self.edges_from = {}
        self.edges_to = {}
    def add_Node(self, node:Node):
        if node.txId not in self.nodes:
            self.nodes[node.txId] = node
            #self.edges[node.txId] = []

    def add_edge(self, current_txId, neighbor_txId):
        self.nodes[current_txId].children.append(neighbor_txId) #add the neighbor to the current node neighbor list
        self.nodes[neighbor_txId].parents.append(current_txId) #add the current node as the node that preceds the neighbor node

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

    def export_graph_nodes(only_labeled=True):
        pass