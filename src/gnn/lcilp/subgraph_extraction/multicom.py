import pandas as pd
import numpy as np
from scipy import sparse
from collections import deque
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

from functools import reduce

from sklearn.cluster import DBSCAN

def load_graph(edgelist_filename, delimiter='\t', comment='#'):
    """
    Load an undirected and unweighted graph from an edge-list file.
    :param edgelist_filename: string or unicode
        Path to the edge-list file.
        Id of nodes are assumed to be non-negative integers.
    :param delimiter: str, default '\t'
    :param comment: str, default '#'
    :return: Compressed Sparse Row Matrix
        Adjacency matrix of the graph
    """
    #edge_df = pd.read_csv(edgelist_filename, delimiter=delimiter, names=["src", "dst"], comment=comment)
    edge_df = pd.read_csv(edgelist_filename, sep=',')
    edge_df = edge_df[["~start_node_id", "~start_node_id"]]
    edge_list = edge_df.values#.as_matrix()
    n_nodes = int(np.max(edge_list) + 1)
    adj_matrix = sparse.coo_matrix((np.ones(edge_list.shape[0]), (edge_list[:, 0], edge_list[:, 1])),                               
                                   shape=tuple([n_nodes, n_nodes]),
                                   dtype=edge_list.dtype)
    adj_matrix = adj_matrix.tocsr()
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix


def convert_adj_matrix(adj_matrix):
    """
    Convert an adjacency matrix to the Compressed Sparse Row type.
    :param adj_matrix: An adjacency matrix.
    :return: Compressed Sparse Row Matrix
        Adjacency matrix with the expected type.
    """
    if type(adj_matrix) == sparse.csr_matrix:
        return adj_matrix
    elif type(adj_matrix) == np.ndarray:
        return sparse.csr_matrix(adj_matrix)
    else:
        raise TypeError("The argument should be a Numpy Array or a Compressed Sparse Row Matrix.")


def approximate_ppr(adj_matrix, seed_set, alpha=0.85, epsilon=1e-3):
    """
    Compute the approximate Personalized PageRank (PPR) from a set set of seed node.
    This function implements the push method introduced by Andersen et al.
    in "Local graph partitioning using pagerank vectors", FOCS 2006.
    :param adj_matrix: compressed sparse row matrix or numpy array
        Adjacency matrix of the graph
    :param seed_set: list or set of int
        Set of seed nodes.
    :param alpha: float, default 0.85
        1 - alpha corresponds to the probability for the random walk to restarts from the seed set.
    :param epsilon: float, default 1e-3
        Precision parameter for the approximation
    :return: numpy 1D array
        Vector containing the approximate PPR for each node of the graph.
    """
    adj_matrix = convert_adj_matrix(adj_matrix)
    degree = np.array(np.sum(adj_matrix, axis=0))[0]
    n_nodes = adj_matrix.shape[0]

    prob = np.zeros(n_nodes)
    res = np.zeros(n_nodes)
    res[list(seed_set)] = 1. / len(seed_set)

    next_nodes = deque(seed_set)

    while len(next_nodes) > 0:
        node = next_nodes.pop()
        push_val = res[node] - 0.5 * epsilon * degree[node]
        res[node] = 0.5 * epsilon * degree[node]
        prob[node] += (1. - alpha) * push_val
        put_val = alpha * push_val
        for neighbor in adj_matrix[node].indices:
            old_res = res[neighbor]
            res[neighbor] += put_val * adj_matrix[node, neighbor] / degree[node]
            threshold = epsilon * degree[neighbor]
            if res[neighbor] >= threshold > old_res:
                next_nodes.appendleft(neighbor)
    return prob


def conductance_sweep_cut(adj_matrix, score, window=10):
    """
    Return the sweep cut for conductance based on a given score.
    During the sweep process, we detect a local minimum of conductance using a given window.
    The sweep process is described by Andersen et al. in
    "Communities from seed sets", 2006.
    :param adj_matrix: compressed sparse row matrix or numpy array
        Adjacency matrix of the graph.
    :param score: numpy vector
        Score used to order the nodes in the sweep process.
    :param window: int, default 10
        Window parameter used for the detection of a local minimum of conductance.
    :return: set of int
         Set of nodes corresponding to the sweep cut.
    """
    adj_matrix = convert_adj_matrix(adj_matrix)
    n_nodes = adj_matrix.shape[0]
    degree = np.array(np.sum(adj_matrix, axis=0))[0]
    total_volume = np.sum(degree)
    sorted_nodes = [node for node in range(n_nodes) if score[node] > 0]
    sorted_nodes = sorted(sorted_nodes, key=lambda node: score[node], reverse=True)
    sweep_set = set()
    volume = 0.
    cut = 0.
    best_conductance = 1.
    best_sweep_set = {sorted_nodes[0]}
    inc_count = 0
    for node in sorted_nodes:
        volume += degree[node]
        for neighbor in adj_matrix[node].indices:
            if neighbor in sweep_set:
                cut -= 1
            else:
                cut += 1
        sweep_set.add(node)

        if volume == total_volume:
            break
        conductance = cut / min(volume, total_volume - volume)
        if conductance < best_conductance:
            best_conductance = conductance
            # Make a copy of the set
            best_sweep_set = set(sweep_set)
            inc_count = 0
        else:
            inc_count += 1
            if inc_count >= window:
                break
    return best_sweep_set


def multicom(adj_matrix, seedset, scoring, cut, explored_ratio=0.8, one_community=True):
    """
    Algorithm for multiple local community detection from a seed node.

    It implements the algorithm presented by Hollocou, Bonald and Lelarge in
    "Multiple Local Community Detection".
    
    Note: Modified by Hebatallah Mohamed 

    :param adj_matrix: compressed sparse row matrix or numpy 2D array
        Adjacency matrix of the graph.
    :param seedset: int (change)
        Id of the seed nodes around which we want to detect communities.
    :param scoring: function
        Function (adj_matrix: numpy 2D array, seed_set: list or set of int) -> score: numpy 1D array.
        Example: approximate_ppr
    :param cut: function
        Function (adj_matrix: numpy 2D array, score: numpy 1D array) -> sweep cut: set of int.
        Example: conductance_sweep_cut
    :param explored_ratio: float, default 0.8
        Parameter used to control the number of new seeds at each step.
    :return:
    seeds: list of int
        Seeds used to detect communities around the initial seed (including this original seed).
    communities: list of set
        Communities detected around the seed node.
    """
   
    scores = []
    communities = list()
    
    #adj_matrix = convert_adj_matrix(adj_matrix)

    if (one_community):
        scores = scoring(adj_matrix, seedset)
        community = cut(adj_matrix, scores)
        communities.append(community)
    else:     
        for seed in seedset:
            scores = scoring(adj_matrix, [seed])
            community = cut(adj_matrix, scores)
            communities.append(community)
    return seedset, communities


def load_groundtruth(groundtruth_filename, delimiter='\t', comment='#'):
    """
    Load a collection of ground-truth communities.

    :param groundtruth_filename: string or unicode
        Path to the file containing the ground-truth.
        Each line of the file must correspond to a community.
        For each line, the ids of the nodes should be integers separated by the specified delimiter.
    :param delimiter: str, default '\t'
    :param comment: str, default '#'
    :return: list of list
        List of ground-truth communities.
    """
    groundtruth_df = pd.read_csv(groundtruth_filename, delimiter="a", names=["list"], comment=comment)
    groundtruth_df.list = groundtruth_df.list.str.split(delimiter)
    groundtruth = [[int(i) for i in row.list] for i, row in groundtruth_df.iterrows()]
    return groundtruth


def extract_subgraph(adj_matrix, groundtruth, nodes):
    """
    Return the subgraph and the ground-truth communities induced by a list of nodes.

    The ids of the nodes of the subgraph are mapped to 0,...,N' - 1 where N' is the length of the list of nodes.

    :param adj_matrix: compressed sparse row matrix or numpy 2D array
        Adjacency matrix of the graph.
    :param groundtruth: list of list/set of int
        List of ground-truth communities
    :param nodes: numpy 1D array
        List of nodes from which we want to extract a subgraph.
    :return:
        new_adj_matrix: compressed sparse row matrix
            Adjacency matrix of the subgraph.
        new_groundtruth: list of list of int
            List of ground-truth communities using the node ids of the subgraph.
        node_map: dictionary
            Map from node ids of the original graph to the ids of the subgraph.
    """
    #adj_matrix = convert_adj_matrix(adj_matrix)
    node_map = {nodes[i]: i for i in range(nodes.shape[0])}
    new_groundtruth = [[node_map[i] for i in community if i in node_map] for community in groundtruth]
    new_adj_matrix = adj_matrix[nodes, :][:, nodes]
    return new_adj_matrix, new_groundtruth, node_map


def get_node_membership(communities):
    """
    Get the community membership for each node given a list of communities.
    :param communities: list of list of int
        List of communities.
    :return: membership: dict (defaultdict) of set of int
        Dictionary such that, for each node,
        membership[node] is the set of community ids to which the node belongs.
    """
    membership = defaultdict(set)
    for i, community in enumerate(communities):
        for node in community:
            membership[node].add(i)
    return membership


def compute_f1_scores(communities, groundtruth):
    """
    Compute the maximum F1-Score for each community of a list of communities
    with respect to a collection of ground-truth communities.

    :param communities: list of list/set of int
        List of communities.
    :param groundtruth: list of list/set of int
        List of ground-truth communities with respect to which we compute the maximum F1-Score.
    :return: f1_scores: list of float
        List of F1-Scores corresponding to the score of each community given in input.
    """
    groundtruth_inv = get_node_membership(groundtruth)
    communities = [set(community) for community in communities]
    groundtruth = [set(community) for community in groundtruth]
    f1_scores = list()
    for community in communities:
        groundtruth_indices = reduce(lambda indices, node: groundtruth_inv[node] | indices, community, set())
        max_precision = 0.
        max_recall = 0.
        max_f1 = 0.
        for i in groundtruth_indices:
            precision = float(len(community & groundtruth[i])) / float(len(community))
            recall = float(len(community & groundtruth[i])) / float(len(groundtruth[i]))
            f1 = 2 * precision * recall / (precision + recall)
            max_precision = max(precision, max_precision)
            max_recall = max(recall, max_recall)
            max_f1 = max(f1, max_f1)
        f1_scores.append(max_f1)
    return f1_scores
