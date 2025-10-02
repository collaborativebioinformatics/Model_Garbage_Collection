import statistics
import numpy as np
import scipy.sparse as ssp
import torch
import networkx as nx
import dgl
import pickle


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ("nodes", "r_label", "g_label", "n_label")
    return dict(zip(keys, data_tuple))


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def incidence_matrix(adj_list):
    """
    adj_list: List of sparse adjacency matrices
    """

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def ssp_to_torch(A, device, dense=False):
    """
    A : Sparse adjacency matrix
    """
    # MIGRATION: Updated for PyTorch 2.1.2/DGL 1.1.3
    idx = torch.tensor([A.tocoo().row, A.tocoo().col], dtype=torch.long)
    dat = torch.tensor(A.tocoo().data, dtype=torch.float)
    # MIGRATION: Updated for PyTorch 2.1.2/DGL 1.1.3
    A = torch.sparse_coo_tensor(idx, dat, size=(A.shape[0], A.shape[1]), device=device)
    return A


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {"type": rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    # MIGRATION: Updated for PyTorch 2.1.2/DGL 1.1.3
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=["type"])
    # add node features
    if n_feats is not None:
        g_dgl.ndata["feat"] = torch.tensor(n_feats)

    return g_dgl


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    (
        graphs_pos,
        g_labels_pos,
        r_labels_pos,
        graphs_negs,
        g_labels_negs,
        r_labels_negs,
    ) = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(graphs_pos)

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

    # Handle case when there are no negative samples (e.g., during scoring)
    if len(graphs_neg) > 0:
        batched_graph_neg = dgl.batch(graphs_neg)
    else:
        batched_graph_neg = None

    return (
        (batched_graph_pos, r_labels_pos),
        g_labels_pos,
        (batched_graph_neg, r_labels_neg),
        g_labels_neg,
    )


def move_batch_to_device_dgl(batch, device):
    ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg) = (
        batch
    )

    # MIGRATION: Updated for PyTorch 2.1.2/DGL 1.1.3
    targets_pos = torch.tensor(targets_pos, dtype=torch.long, device=device)
    r_labels_pos = torch.tensor(r_labels_pos, dtype=torch.long, device=device)

    # Handle case when there are no negative samples
    if g_dgl_neg is not None:
        targets_neg = torch.tensor(targets_neg, dtype=torch.long, device=device)
        r_labels_neg = torch.tensor(r_labels_neg, dtype=torch.long, device=device)
        g_dgl_neg = send_graph_to_device(g_dgl_neg, device)
    else:
        targets_neg = torch.tensor([], dtype=torch.long, device=device)
        r_labels_neg = torch.tensor([], dtype=torch.long, device=device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)

    return (
        (g_dgl_pos, r_labels_pos),
        targets_pos,
        (g_dgl_neg, r_labels_neg),
        targets_neg,
    )


def send_graph_to_device(g, device):
    # nodes
    # MIGRATION: Updated for PyTorch 2.1.2/DGL 1.1.3
    for key in g.ndata.keys():
        g.ndata[key] = g.ndata[key].to(device)

    # edges
    # MIGRATION: Updated for PyTorch 2.1.2/DGL 1.1.3
    for key in g.edata.keys():
        g.edata[key] = g.edata[key].to(device)
    return g


#  The following three functions are modified from networks source codes to
#  accomodate diameter and radius for dirercted graphs


def eccentricity(G):
    e = {}
    # MIGRATION: Updated for PyTorch 2.1.2/DGL 1.1.3
    for n in G.nodes():
        length = nx.single_source_shortest_path_length(G, n)
        e[n] = max(length.values())
    return e


def radius(G):
    e = eccentricity(G)
    e = np.where(np.array(list(e.values())) > 0, list(e.values()), np.inf)
    return min(e)


def diameter(G):
    e = eccentricity(G)
    return max(e.values())
