from dataclasses import dataclass
from typing import Optional, List, Tuple, Set
from src.gnn.lcilp.utils.data_utils import process_files
from src.gnn.lcilp.subgraph_extraction.graph_sampler import links2subgraphs
from src.gnn.lcilp.utils.graph_utils import deserialize
import argparse
import os
import lmdb
import numpy as np

# locate graph files associated with their name
# NOTE: cannot change name of 'train'
files = {
    'train': './lcilp/data/alzheimers_triples.txt',
    'query_backbone': './lcilp/data/alzheimers_backbone_triples.txt'
}


@dataclass
class Params:
    """Parameters for subgraph extraction"""
    db_path: str
    hop: int # k-hop neighborhood
    enclosing_sub_graph: bool  # Extract enclosing subgraph
    max_nodes_per_hop: Optional[int]  # No limit on nodes per hop
    map_size_multiplier: int


def load_subgraphs_from_db(db_path: str, split_name: str = 'query') -> List[dict]:
    """
    Load extracted subgraphs from LMDB database.

    Args:
        db_path: Path to LMDB database directory
        split_name: Name of the split (e.g., 'query', 'train', 'valid')

    Returns:
        List of subgraph dictionaries with 'nodes', 'r_label', 'g_label', etc.
    """
    env = lmdb.open(db_path, readonly=True, max_dbs=6, lock=False)
    db_name_pos = f'{split_name}_pos'
    db_pos = env.open_db(db_name_pos.encode())

    subgraphs = []
    with env.begin(db=db_pos) as txn:
        # Get number of graphs
        num_graphs = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        # Load each subgraph
        for idx in range(num_graphs):
            str_id = '{:08}'.format(idx).encode('ascii')
            datum = deserialize(txn.get(str_id))
            subgraphs.append(datum)

    env.close()
    return subgraphs


def extract_edges_from_subgraphs(subgraphs: List[dict], adj_list: List) -> List[Tuple[int, int, int]]:
    """
    Extract all unique edges from subgraphs using original adjacency matrices.

    Args:
        subgraphs: List of subgraph dictionaries with 'nodes' field
        adj_list: List of scipy sparse adjacency matrices (one per relation)

    Returns:
        List of unique (head_id, tail_id, relation_id) tuples
    """
    all_edges = set()

    for subgraph in subgraphs:
        nodes = subgraph['nodes']

        # For each relation type
        for rel_id, adj_matrix in enumerate(adj_list):
            # Extract submatrix for these nodes
            submatrix = adj_matrix[nodes, :][:, nodes]

            # Find all edges in this submatrix
            rows, cols = submatrix.nonzero()

            # Map back to original node IDs
            for i, j in zip(rows, cols):
                head = nodes[i]
                tail = nodes[j]
                all_edges.add((head, tail, rel_id))

    return list(all_edges)


def convert_ids_to_labels(edges: List[Tuple[int, int, int]],
                          id2entity: dict,
                          id2relation: dict) -> List[Tuple[str, str, str]]:
    """
    Convert edge IDs to human-readable labels.

    Args:
        edges: List of (head_id, tail_id, relation_id) tuples
        id2entity: Mapping from entity ID to entity name
        id2relation: Mapping from relation ID to relation name

    Returns:
        List of (subject, predicate, object) string tuples
    """
    labeled_edges = []

    for head_id, tail_id, rel_id in edges:
        subject = id2entity[head_id]
        predicate = id2relation[rel_id]
        obj = id2entity[tail_id]
        labeled_edges.append((subject, predicate, obj))

    return labeled_edges


def export_triples(labeled_edges: List[Tuple[str, str, str]],
                   output_path: str,
                   format: str = 'tsv'):
    """
    Export labeled triples to file.

    Args:
        labeled_edges: List of (subject, predicate, object) tuples
        output_path: Where to save the file
        format: Output format ('tsv', 'csv', or 'nt')
    """
    delimiter = '\t' if format == 'tsv' else ','

    with open(output_path, 'w') as f:
        for subject, predicate, obj in labeled_edges:
            if format == 'nt':
                # N-Triples RDF format
                f.write(f"<{subject}> <{predicate}> <{obj}> .\n")
            else:
                f.write(f"{subject}{delimiter}{predicate}{delimiter}{obj}\n")


def merge_and_export_subgraphs(params: Params,
                               adj_list: List,
                               entity2id: dict,
                               relation2id: dict,
                               id2entity: dict,
                               id2relation: dict,
                               output_path: str) -> int:
    """
    Merge all extracted subgraphs and export as labeled triples.

    Args:
        params: Params object with db_path
        adj_list: List of adjacency matrices (one per relation)
        entity2id: Entity name → ID mapping
        relation2id: Relation name → ID mapping
        id2entity: ID → Entity name mapping
        id2relation: ID → Relation name mapping
        output_path: Where to save merged triples

    Returns:
        Number of unique edges exported
    """
    # Step 1: Load subgraphs from database
    print(f"\nLoading subgraphs from {params.db_path}...")
    subgraphs = load_subgraphs_from_db(params.db_path, split_name='query')
    print(f"Loaded {len(subgraphs)} subgraphs")

    # Step 2: Extract all unique edges
    print("Extracting edges from subgraphs...")
    edges = extract_edges_from_subgraphs(subgraphs, adj_list)
    print(f"Found {len(edges)} unique edges")

    # Step 3: Convert IDs to labels
    print("Converting IDs to labels...")
    labeled_edges = convert_ids_to_labels(edges, id2entity, id2relation)

    # Step 4: Export to file
    print(f"Exporting to {output_path}...")
    export_triples(labeled_edges, output_path, format='tsv')

    print(f"✓ Exported {len(labeled_edges)} triples to {output_path}")
    return len(labeled_edges) 


if __name__ == '__main__':
    # read the whole graph (big G) and the backbone edges (E^{q}_{b})
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(files)

    # there is a single graph object: G which combines all of the files to load
    # triplets contains namespaced set of edges from each file
    # Note: adj_list is built only from 'train' data (see data_utils.py:60)

    # Setup parameters for subgraph extraction
    params = Params(db_path='./lcilp/data/subgraphs_db', 
                    hop=3, 
                    enclosing_sub_graph=True, 
                    max_nodes_per_hop=None,
                    map_size_multiplier=10)

    # Create output directory for LMDB database
    os.makedirs(params.db_path, exist_ok=True)

    # Prepare graphs dict for links2subgraphs
    # For query_backbone edges, we treat them all as positive samples
    graphs = {
        'query': {
            'pos': triplets['query_backbone'],  # Backbone edges as positive
            'neg': []  # No negative samples for now
        }
    }

    # Extract subgraphs for query_backbone edges
    print(f"Extracting subgraphs for {len(triplets['query_backbone'])} backbone edges...")
    links2subgraphs(adj_list, graphs, params, max_label_value=None)
    print(f"Subgraphs saved to {params.db_path}")

    # Merge and export subgraphs
    output_file = './lcilp/data/alzheimers_merged_subgraphs.txt'
    num_edges = merge_and_export_subgraphs(
        params, adj_list, entity2id, relation2id,
        id2entity, id2relation, output_file
    ) 