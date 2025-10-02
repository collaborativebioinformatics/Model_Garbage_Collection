from dataclasses import dataclass
from typing import Optional
from src.gnn.lcilp.utils.data_utils import process_files
from src.gnn.lcilp.subgraph_extraction.graph_sampler import links2subgraphs
import argparse
import os

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