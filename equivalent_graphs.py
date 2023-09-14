import matplotlib as plt
import networkx as nx
import itertools
from itertools import permutations


def ordered_pairs_of_numbers(n): # generates all ordered pairs of numbers between 0 and n-1
    # Create a list of numbers from 0 to n-1
    numbers = list(range(n))
    
    # Generate all ordered pairs (tuples of size 2) of the numbers
    ordered_pairs = [(x, y) for x in numbers for y in numbers if y != x]
    
    return ordered_pairs

def all_subsets(elements): # recursively generates all possible subsets of a list
    if not elements:
        return [[]]  # The empty set is always a subset

    subsets_without_first = all_subsets(elements[1:])
    subsets_with_first = [[elements[0]] + subset for subset in subsets_without_first]

    return subsets_without_first + subsets_with_first

def has_duplicate_sets(Y): # checks whether in an edgelist a bidirectional edge is contained.
    lst=[]
    for y in Y:
        lst.append(set(y))
    seen = []
    for s in lst:
        if s in seen:
            return True
        seen.append(s)
    return False

def avoid_bidirect(X): # only keeps edge lists which have no bidirectional edges.
    mylist=[]
    for x in X:
        if has_duplicate_sets(x):
            continue
        else:
            mylist.append(x)

    return mylist

def create_permutations(n): # generates all possible permutations of n elements through a mapping dictionary
    numbers = list(range(n))
    # Generate all permutations
    permutations = list(itertools.permutations(numbers))
    mappings = []
    for perm in permutations:
        mapping = {}
        for i in range(n):
            mapping[i]=perm[i]
        mappings.append(mapping)
    return mappings

def two_graphs_equivalent(graph_1,graph_2,mappings): # calculates when two graphs are equivalent
    for mapping in mappings:
        temp_graph_1 = [(mapping.get(u, u), mapping.get(v, v)) for u, v in graph_1] # map graph_1 through mapping
        if set(temp_graph_1)==set(graph_2):
            return True
    return False

def remove_equivalent(graphs,n): # from a list of graphs this function only keeps one representative per equivalence class
    mappings=create_permutations(n)
    new_graphs=[]
    viewed_graphs=[]
    for graph_1 in graphs:
        for graph_2 in graphs:
            if graph_1 == graph_2:
                continue
            elif graph_1 in viewed_graphs or graph_2 in viewed_graphs:
                if two_graphs_equivalent(graph_1,graph_2,mappings):
                    if graph_1 not in viewed_graphs:
                        viewed_graphs.append(graph_1)
                    elif graph_1 not in viewed_graphs:
                        viewed_graphs.append(graph_2)
                    continue
            else:
                if two_graphs_equivalent(graph_1,graph_2,mappings):
                    new_graphs.append(graph_1)
                    viewed_graphs.append(graph_1)
                    viewed_graphs.append(graph_2)

    return new_graphs



### --------------------------------------------------------------

if __name__ == "__main__":
    
    n = 4
    all_ordered_pairs = ordered_pairs_of_numbers(n)
    bidirect_graphs = all_subsets(all_ordered_pairs)
    duplicate_graphs = avoid_bidirect(bidirect_graphs)
    graphs=remove_equivalent(duplicate_graphs,n)

    print(graphs)