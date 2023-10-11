
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

    def __init__(self, feats) -> None:
        self.features:list = feats
        self.children = []
        self.parents = []
        self.txId = int(feats[0]) #the first element of features is always the txId
        self.timestep = int(feats[1]) #the second element in always the timestep.
        self.label = int(feats[-1]) 
    
    
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
        edge_pairs = []

        if len(self.children) != 0:
            for childTxId in self.children:
                edge_pairs.append((self.txId, childTxId))
        
        return edge_pairs
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
        self.node_txId_set = set()
    def add_Node(self, node:Node):
        if node.txId not in self.nodes:
            self.nodes[node.txId] = node
            self.node_txId_set.add(node.txId)
            #self.edges[node.txId] = []
    
    def create_from_lists(self, features, edge_list):
        """
            Creates a graph object from a list of features and a list of edges.
            :attribute features: A list with the node features. A merge with a pandas dataframe has first occured so that it also includes the class in the list's last index.
            :attribute features: A list with node edges.

        """
        for row in features:
            txId = int(row[0])
            if txId not in self.nodes:
                self.add_Node(Node(row))

        for edge in edge_list:
            curr_txId = int(edge[0])
            neighbor = int(edge[1])
            if curr_txId in self.nodes:
                self.add_edge(curr_txId, neighbor)

    def add_edge(self, current_txId, neighbor_txId):
        """
            Adds a node in the graphs and creates the associations between its neighbors
            :attribute current_txId: The txId of the current node (int).
            :attribute neighbor_txId: The txId of the neighbor_txId node (int).

        """
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

    
                            
