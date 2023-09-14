import networkx as nx
import random as rd
import math
import matplotlib.pyplot as plt
import copy
import pyflagser


### -------------network-initializations--------------------------
def initialise_node_attributes(G):
    for node in G.nodes():
        G.nodes[node]['state'] = 0
        G.nodes[node]['virtual'] = False
        G.nodes[node]['predecessors'] = list(G.predecessors(node))
        G.nodes[node]['neighbor_states'] = []
        G.nodes[node]['update_rule'] = and_update
    return

def and_update(neighbor_states,state): # standard update rule for neurons
    if neighbor_states == []:
        return 0
    if all(neighbor_state == 1 for neighbor_state in neighbor_states):
        return 1 
    return 0

def keep_state_update(neighbor_states,state): # virtual neuron's update rule
    return state

def update_inc_states(G): # updates the incoming states of all nodes
    for node in G.nodes():
        G.nodes[node]['neighbor_states'] = []
        for pred in G.nodes[node]['predecessors']:
            G.nodes[node]['neighbor_states'].append(G.nodes[pred]['state']) 
    return

def update_states(G): # updates the states of all nodes depending on their incoming states
    for node in G.nodes():
        G.nodes[node]['state'] = G.nodes[node]['update_rule'](G.nodes[node]['neighbor_states'],G.nodes[node]['state'])
    return

def update(G): # one update step of the system
    update_inc_states(G)
    update_states(G)
    return

def init_config(G,config): # initialises the system with a given configuration config
    if len(config) != len(G.nodes()):
        print('Error: Config length unequal number of nodes')
        return
    k=0
    for i in G.nodes():
        G.nodes[i]['state']=config[k]
        k+=1
    return

def curr_config(G): # returns current system configuration
    config=[]
    for node in G.nodes():
        config.append(G.nodes[node]['state'])
    return tuple(config)

def get_virtual_nodes(G): # returns all virtual nodes of the system
    virtual_nodes=[]
    for node in G.nodes():
        if G.nodes[node]['virtual']:
            virtual_nodes.append(node)
    return virtual_nodes
### --------------------------------------------------------------

### -------------partitioning-DiGraph-----------------------------
def check_virtual_node(subgraph,part,node): # check if virtual node has to be added
    outliers=[]
    if subgraph.nodes[node]['predecessors']!=[]:
        for pred in subgraph.nodes[node]['predecessors']:
            if pred not in part:
                outliers.append(pred)
    return outliers

def add_virtual_node(subgraph,outliers,node): # adds a virtual node and properly initialises it
    k = max(99, max(subgraph.nodes())) + 1 # virtual nodes are numbered from 100 on
    subgraph.add_edge(k,node)
    
    # update predecessors of node to not contain outliers except for one
    subgraph.nodes[node]['predecessors'] = [item for item in subgraph.nodes[node]['predecessors'] if item not in outliers]
    subgraph.nodes[node]['predecessors'].append(k)

    # update the new node to be a virtual node with the correct update rule
    subgraph.nodes[k]['virtual'] = True
    subgraph.nodes[k]['state'] = 0
    subgraph.nodes[k]['predecessors'] = [] 
    subgraph.nodes[k]['neighbor_states'] = []
    subgraph.nodes[k]['update_rule'] = keep_state_update
    return

def partitioned_subgraphs(G,partition): # partitions a system according to a parition by calling the partition function for each part
    subgraphs=[]
    for part in partition:
        subgraphs.append(partitioned_subgraph(G,part))
    return subgraphs

def partitioned_subgraph(G,part): # adds virtual nodes according to the definitions of a subsystem
    temp_subgraph = G.subgraph(part)
    subgraph = temp_subgraph.copy()
    for node in temp_subgraph.nodes():
        outliers = check_virtual_node(subgraph,part,node)
        if outliers !=[]:
            add_virtual_node(subgraph,outliers,node)
    return subgraph
### --------------------------------------------------------------

### ----------calculate-EI/P--------------------------------------
def EI(G): # calculates total effective information for all partitions
    total_distr = target_distr(G)
    effective_info = {}
    for partition in partitions(G.number_of_nodes()):
        parts_distr = EI_P(G,partition) # contains the probability distributions of the parts
        parts_distr.insert(0, partition) # the first element of the distribution list is the partition
        effective_info[tuple(tuple(part) for part in partition)] = calc_EI_P(total_distr, parts_distr) 
    return effective_info

def EI_P(G,partition): # calculates the distributions of configurations of the parts in the partition
    parts_distr=[]
    subgraphs = partitioned_subgraphs(G,partition)
    for subgraph in subgraphs:
        parts_distr.append(target_distr(subgraph)) # M_0 restricted to preimage of targets
    return parts_distr

def calc_EI_P(total_distr, parts_distr): # calculates KL-divergence of the c(f()) vs f(c())
    KL = 0
    
    parts_distr=collective_distribution(parts_distr)
    
    total_distr = prob_conversion(total_distr)
    parts_distr = prob_conversion(parts_distr)
    return KLdiv(total_distr, parts_distr)

def collective_distribution(parts_distr): # this is function c. it concatenates all the subconfigurations in the correct order
    partition=parts_distr.pop(0)
    length=0 # gives length of total configuration
    for d in parts_distr:
        length+=len(list(d.keys())[0])
    total_configs=[[]]
    result={}
    for k in range(length): # this list will be duplicated and filled in in the recursion steps. It stores total configs.
        total_configs[0].append(-1)

    if partition == [] or parts_distr == []:
        print('error')
        return
    
    amounts=[1]
    j=0
    # geh in erstes dictionary von parts distribution
    for part_distr in parts_distr:
        new_total_configs=[]
        
        for key in part_distr:
            #  für jeden key tu folgendes:
            temp_total_configs=copy.deepcopy(total_configs)
            temp_key = list(key)
            
            k=0
            for state in temp_key:
                for i in range(len(total_configs)):
                    # inserte in eine temporäre kopie der total distributions, diese keys entsprechend ihrer position in partition
                    temp_total_configs[i][partition[j][k]]=state
                k+=1
            new_total_configs+=temp_total_configs
            
        new_amounts = []
        for key in part_distr:
            for amount in amounts: 
                new_amounts.append(part_distr[key]*amount)
        j+=1
        total_configs=copy.deepcopy(new_total_configs)
        amounts=copy.deepcopy(new_amounts)
    
    result = {} # results to dictionary
    for i in range(len(amounts)):
        result[tuple(total_configs[i])]=amounts[i]
    return result

def prob_conversion(amount_dict):  # turn dict with amounts into probabilities
    total=sum(amount_dict.values())
    for x in amount_dict: 
        amount_dict[x] = amount_dict[x]/total
    return amount_dict

def KLdiv(P,Q): # calculates KL-divergence of two probability distributions
    KL = 0
    for omega in P:
        if P[omega] == 0:
            continue
        if omega not in Q:
            continue
        if Q[omega] == 0:
            continue
        KL += P[omega] * math.log(P[omega]/Q[omega],2) # well defined, as Q(x) = 0 implies P(x) = 0
    return KL

def target_distr(G): # calculate probabilities of configurations in G after one update.
    target_distr={}
    virtual_nodes = get_virtual_nodes(G)
    virtual_nodes_indices = [list(G.nodes()).index(node) for node in virtual_nodes]
    configs=generate_all_configs(G.number_of_nodes())
    for config in configs:
        init_config(G,config)
        update(G)
        temp_config = curr_config(G)
        # get rid of virtual states for comparison
        temp_config = tuple(element for i, element in enumerate(temp_config) if i not in virtual_nodes_indices)
        if temp_config in target_distr: 
            target_distr[temp_config] += 1
        else:
            target_distr[temp_config] = 1
    return target_distr
### --------------------------------------------------------------

### ----------genereate-all-partitions----------------------------
def partition_recursion(n, k, partitions): # recursively generates all possible partitions
    if k == n:
        return partitions
    new_partitions = []
    for parts in partitions:
        for i in range(len(parts)):
            temp_parts = copy.deepcopy(parts)  # Create a real copy of the partition
            temp_parts[i].append(k)
            new_partitions.append(temp_parts)
        parts.append([k])  # Create a new subset containing only the current element k
        new_partitions.append(parts)
    return partition_recursion(n, k + 1, new_partitions)  # Increment k by 1

def partitions(n): # return all covers of a list of integers 0,...,n-1
    result = partition_recursion(n, 0, [[]])
    return result
### --------------------------------------------------------------

### ----------all-possible-configs--------------------------------
def generate_all_configs(n, word=tuple()): # generates all possible configurations of a system recursively
    if n == 0:
        return [word]
    return generate_all_configs(n - 1, word + (0,)) + generate_all_configs(n - 1, word + (1,))
### --------------------------------------------------------------

### -----------network-examples-----------------------------------
def random_network(num_nodes,edge_probability):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    # Add edges with the specified probability
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2 and rd.random() < edge_probability:
                G.add_edge(node1, node2)
    return G

def fully_connected(n): # fully connected with three nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                G.add_edge(node1, node2)
    return G

def directed_cycle(n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for node in range(G.number_of_nodes()):
        G.add_edge(node, node+1)
    G.add_edge(n, 0)
    return G

def bidirectional_cycle(n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for node in range(G.number_of_nodes()):
        G.add_edge(node, node+1)
        G.add_edge(node+1, node)
    G.add_edge(n, 0)
    G.add_edge(0, n)
    return G

def wedge_central_out(n):
    # Create an empty directed graph
    G = nx.DiGraph()
    for i in range(n):
        cycle_nodes = ['cycle{}_{}'.format(i, j) for j in range(3)]
        G.add_nodes_from(cycle_nodes)
        G.add_edges_from([(cycle_nodes[j], cycle_nodes[(j + 1) % 3]) for j in range(3)])

    # Add a central neuron connected to one neuron in each cycle
    central_neuron = 'central'
    G.add_node(central_neuron)

    for i in range(n):
        cycle_nodes = ['cycle{}_{}'.format(i, j) for j in range(3)]
        G.add_edge(central_neuron, cycle_nodes[0])
    return G

def wedge_central_in(n):
    # Create an empty directed graph
    G = nx.DiGraph()
    for i in range(n):
        cycle_nodes = ['cycle{}_{}'.format(i, j) for j in range(3)]
        G.add_nodes_from(cycle_nodes)
        G.add_edges_from([(cycle_nodes[j], cycle_nodes[(j + 1) % 3]) for j in range(3)])

    # Add a central neuron connected to one neuron in each cycle
    central_neuron = 'central'
    G.add_node(central_neuron)

    for i in range(n):
        cycle_nodes = ['cycle{}_{}'.format(i, j) for j in range(3)]
        G.add_edge( cycle_nodes[0], central_neuron) # changed order compared to above
    return G

def wedge_node(n):
    # Create an empty directed graph
    G = nx.DiGraph()
    for i in range(n):
        cycle_nodes = ['cycle{}_{}'.format(i, j) for j in range(3)]
        H=nx.DiGraph()
        H.add_nodes_from(cycle_nodes)
        H.add_edges_from([(cycle_nodes[j], cycle_nodes[(j + 1) % 3]) for j in range(3)])
        H = nx.relabel_nodes(H, {cycle_nodes[0]:0})
        G.update(H)
    return G

def wedge_edge(n): # how to do this?
    # Create an empty directed graph
    G = nx.DiGraph()
    G.add_edges_from([(0,1),(1,2),(0,2),()])
    return G

### --------------------------------------------------------------


### -----------general-functions----------------------------------
def plot_graph(graph):
    layout = nx.planar_layout(graph)
    plt.figure()
    nx.draw(graph, pos=layout, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, arrows=True)
    return

def total_int_info(G):
    initialise_node_attributes(G)
    my_dict=EI(G)
    max_value = max(my_dict.values())
    #max_keys = [key for key, value in my_dict.items() if value == max_value]

    #print("Key(s):", max_keys)
    #print("Value:", max_value)  
    
    A=nx.adjacency_matrix(G)
    x=pyflagser.flagser_unweighted(A, coeff=17)

    #plot_graph(G)
    return [max_value,x['betti']]

def ordered_pairs_of_numbers(n):
    # Create a list of numbers from 0 to n-1
    numbers = list(range(n))
    
    # Generate all ordered pairs (tuples of size 2) of the numbers
    ordered_pairs = [(x, y) for x in numbers for y in numbers if y != x]
    
    return ordered_pairs

def all_subsets(elements):
    if not elements:
        return [[]]  # The empty set is always a subset

    subsets_without_first = all_subsets(elements[1:])
    subsets_with_first = [[elements[0]] + subset for subset in subsets_without_first]

    return subsets_without_first + subsets_with_first


### --------------------------------------------------------------

if __name__ == "__main__":
    
    G=random_network(4,0.3)
    print(total_int_info(G))
