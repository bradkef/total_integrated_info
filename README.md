# total_integrated_info
This package computes total effective information of all possible partitions of a system.

To reproduce the code execute following: 
1. pip install -r "requirements.txt"
2. python3 total_integrated_info.py 

A system is a network which has nodes that can attain values 0 or 1 and which have update rules according to which these states change over time. The states of a system together at a given time t are a configuration.

We present the four main parts of the code here.

---

# System representation

To represent the system, we define attributes on the networkx graph nodes. In `network-initializations` functions needed to initialize and update the system can be found.
We show the most important fucntions here.

## Functions

### `initialise_node_attributes(G)`

This function initializes node attributes within the network `G`. It assigns the following attributes to each node:
- `state`: Initialized to `0`, representing the node's current state.
- `virtual`: Initialized to `False`, indicating whether the node is virtual (`True` for virtual nodes, `False` for regular nodes).
- `predecessors`: A list of the node's predecessors within the network.
- `neighbor_states`: An empty list, which will store the states of neighboring nodes.
- `update_rule`: Initialized to the `and_update` function, defining the update of configurations. Can be adapted to any update rule.

### `update(G)`

A convenience function that updates both the `neighbor_states` and node states using the `update_inc_states` and `update_states` functions, respectively.

### `update_inc_states(G)`

This function updates the `neighbor_states` attribute for all nodes in the network. It calculates the states of neighboring nodes and stores them in the `neighbor_states` attribute.

### `update_states(G)`

Updates the states of all nodes in the network based on their respective update rules and the calculated `neighbor_states`.

---

# Partitioning systems

In `partitioning-Digraph` the necessary functions to partition a system are collected. From there, the graph is first restricted and then augmented again by adding virtual nodes when there were incoming edges.

## Functions

### `partitioned_subgraphs(G, partition)`

This function partitions the input system `G` into multiple subsystems based on the given partition. It returns a list of subgraphs, each corresponding to one partition in the input.

### `partitioned_subgraph(G, part)`

Given a partition `part`, this function generates a subgraph from the input graph `G`. It restricts the subgraph to nodes that belong to the specified partition. If there are nodes with predecessors outside the partition detected within the subgraph, it uses the `add_virtual_node` function to add virtual nodes where needed.

### `add_virtual_node(subgraph, outliers, node)`

Adds a virtual node to the subgraph when there are outliers detected by the `check_virtual_node` function. The virtual node is assigned a unique identifier and connected to the original node. Predecessors of the original node that are outliers are updated to exclude these outliers except for the newly added virtual node. The virtual node is configured with specific attributes, making it suitable for use as a virtual node.

### `check_virtual_node(subgraph, part, node)`

This function checks if a virtual node needs to be added to the subgraph based on the given partition. It looks at the predecessors of the specified node within the subgraph and identifies any predecessors that are not part of the partition. If such predecessors exist, they are considered "outliers" which trigger the addition of a virtual node.

---

# Generation of partitions and configurations

These functions provide utilities for generating partitions and configurations from the number of neurons `n`. These functions produce the necessary partitions and configurations to use the above functions.

## Functions

### `partitions(n)`

Generates all possible partitions of a list of integers from `0` to `n-1`. The result is a list of partitions, where each partition is represented as a list of indices.

### `partition_recursion(n, k, partitions)`

A helper function used in `partitions(n)` for recursively generating partitions. It takes `n`, the current element `k`, and a list of partitions as input and returns the new list of partitions, where the currect element has been added to all possible parts that it could belong to.

### `generate_all_configs(n, word=tuple())`

Generates all possible configurations of `n` elements recursively. Each element can take on two values, `0` or `1`. The function returns a list of tuples, where each tuple represents a unique configuration.

---

# Calculation of Total Integrated Information

To calcuulate the total integrated information, we rely on different functions, which calculate total effective information of subparts.

## Functions

### `total_int_info(G)`

This function calculates the total integrated information of a network `G`. It utilizes the following steps to generate relevant information:

1. **Attribute Initialization:** The function begins by initializing various attributes for nodes within the network using the `initialise_node_attributes(G)` function. This step sets up the necessary attributes as discussed before.

2. **Effective Information (EI) Calculation:** The `EI(G)` function is then called to compute the effective information of the network. Effective information is calculated by exploring different partitions of the network's nodes and measuring the Kullback-Leibler (KL) divergence of configurations within these partitions. The result is a dictionary where each key corresponds to a partition, and the value represents the calculated effective information.

3. **Max Effective Information:** Identifies the maximum effective information value from the computed results, providing insight into the partition or configuration that maximizes the information content within the network.

4. **Network Structure Analysis:** In addition to effective information, the function also conducts a structural analysis of the network. It uses the `pyflagser.flagser_unweighted` function to calculate Betti numbers of the associated directed flag complex.

5. **Return Values:** The function returns a list containing two elements: the maximum effective information value and the network's Betti numbers.

### `EI(G)`

This function calculates effective information for a given system `G`. It explores the partitions of the network's nodes and calculates the Kullback-Leibler (KL) divergence of configurations within these partitions. The results are returned as a dictionary, where each key corresponds to a partition, and the value represents the calculated effective information.

### `EI_P(G, partition)`

Calculates the probability distributions of configurations within the parts of a partition in the system `G`. It initializes all possible configurations and updates them on the subsystems. The result is a list of probability distributions for each part.

### `calc_EI_P(total_distr, parts_distr)`

Calculates the KL-divergence of the collective distribution between the total distribution `f(c())` and the parts `c(f())`. `total_distr` represents the total distribution of configurations, and `parts_distr` represents the probability distributions of configurations within partitions.

### `collective_distribution(parts_distr)`

This is function `c`. It generates the collective distribution of configurations across partitions. It combines individual probability distributions within partitions to create a comprehensive distribution of all configurations.

### `prob_conversion(amount_dict)`

Converts a dictionary of amounts into probabilities. Normalizes the values in the dictionary to represent probabilities.

### `KLdiv(P, Q)`

Calculates the Kullback-Leibler divergence between two probability distributions, `P` and `Q` encoded as dictionaries.

### `target_distr(G)`

Calculates the probabilities of configurations in the network `G` after one update. It considers the network's target distribution and explores configurations by updating the network and counting occurrences.

---

Further, `network-examples`, different small examples of networks, such as random networks, cycles, fully connected networks and types of wedges of cycles are provided, which can be used for initialization.


## Acknowledgements

A big thanks to Marc for his valuable ideas on representing systems in python! https://github.com/marcsinner
