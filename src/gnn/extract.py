import sys
import os

# Add lcilp directory to path for relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lcilp"))

from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict
from utils.data_utils import process_files
from subgraph_extraction.graph_sampler import links2subgraphs
from utils.graph_utils import deserialize
import argparse
import lmdb
import numpy as np
import random

# locate graph files associated with their name
# NOTE: cannot change name of 'train'
files = {
    "train": "./lcilp/data/alzheimers_triples.txt",
    "query_backbone": "./lcilp/data/alzheimers_backbone_triples.txt",
}

# Output directory for focused dataset
FOCUSED_OUTPUT_DIR = "./lcilp/data/AlzheimersKG/focused"


@dataclass
class Params:
    """Parameters for subgraph extraction"""

    db_path: str
    hop: int  # k-hop neighborhood
    enclosing_sub_graph: bool  # Extract enclosing subgraph
    max_nodes_per_hop: Optional[int]  # No limit on nodes per hop
    map_size_multiplier: int


def load_subgraphs_from_db(db_path: str, split_name: str = "query") -> List[dict]:
    """
    Load extracted subgraphs from LMDB database.

    Args:
        db_path: Path to LMDB database directory
        split_name: Name of the split (e.g., 'query', 'train', 'valid')

    Returns:
        List of subgraph dictionaries with 'nodes', 'r_label', 'g_label', etc.
    """
    env = lmdb.open(db_path, readonly=True, max_dbs=6, lock=False)
    db_name_pos = f"{split_name}_pos"
    db_pos = env.open_db(db_name_pos.encode())

    subgraphs = []
    with env.begin(db=db_pos) as txn:
        # Get number of graphs
        num_graphs = int.from_bytes(txn.get("num_graphs".encode()), byteorder="little")

        # Load each subgraph
        for idx in range(num_graphs):
            str_id = "{:08}".format(idx).encode("ascii")
            datum = deserialize(txn.get(str_id))
            subgraphs.append(datum)

    env.close()
    return subgraphs


def extract_edges_from_subgraphs(
    subgraphs: List[dict], adj_list: List
) -> List[Tuple[int, int, int]]:
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
        nodes = subgraph["nodes"]

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


def convert_ids_to_labels(
    edges: List[Tuple[int, int, int]], id2entity: dict, id2relation: dict
) -> List[Tuple[str, str, str]]:
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


def export_triples(
    labeled_edges: List[Tuple[str, str, str]], output_path: str, format: str = "tsv"
):
    """
    Export labeled triples to file.

    Args:
        labeled_edges: List of (subject, predicate, object) tuples
        output_path: Where to save the file
        format: Output format ('tsv', 'csv', or 'nt')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    delimiter = "\t" if format == "tsv" else ","

    with open(output_path, "w") as f:
        for subject, predicate, obj in labeled_edges:
            if format == "nt":
                # N-Triples RDF format
                f.write(f"<{subject}> <{predicate}> <{obj}> .\n")
            else:
                f.write(f"{subject}{delimiter}{predicate}{delimiter}{obj}\n")


def load_triplets_from_file(file_path: str) -> List[Tuple[str, str, str]]:
    """
    Load triplets from TSV file.

    Args:
        file_path: Path to TSV file

    Returns:
        List of (subject, predicate, object) tuples
    """
    triplets = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    triplets.append(tuple(parts))
    return triplets


def prepare_focused_dataset(
    merged_file: str,
    backbone_file: str,
    output_dir: str = FOCUSED_OUTPUT_DIR,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Prepare focused dataset with backbone isolation.

    This function:
    1. Loads merged subgraphs and backbone edges
    2. Removes backbone contamination (data leakage prevention)
    3. Splits context edges into train/valid/test
    4. Validates no overlap between splits
    5. Exports files to output directory

    Args:
        merged_file: Path to merged subgraphs file (222 edges)
        backbone_file: Path to backbone edges file (29 edges)
        output_dir: Directory to save split files
        train_ratio: Proportion for training set (default: 0.70)
        valid_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'valid', 'test', 'validation_pool' edge lists
    """
    print("=" * 60)
    print("Preparing Focused Dataset for HITL Training")
    print("=" * 60)

    # Step 1: Load files
    print("\n[1/5] Loading files...")
    merged_edges = load_triplets_from_file(merged_file)
    backbone_edges = set(load_triplets_from_file(backbone_file))

    print(f"  - Merged file: {len(merged_edges)} edges")
    print(f"  - Backbone file: {len(backbone_edges)} edges")

    # Step 2: Remove backbone contamination
    print("\n[2/5] Removing backbone contamination...")
    context_edges = [e for e in merged_edges if e not in backbone_edges]

    print(f"  - Context edges (merged - backbone): {len(context_edges)} edges")
    print(f"  - Backbone edges isolated for human review: {len(backbone_edges)} edges")

    # Validation: Check if all backbone edges were in merged file
    backbone_in_merged = sum(1 for e in backbone_edges if e in merged_edges)
    print(
        f"  - Backbone edges found in merged: {backbone_in_merged} / {len(backbone_edges)}"
    )

    if backbone_in_merged != len(backbone_edges):
        print(f"  ⚠️  WARNING: Not all backbone edges found in merged file!")

    # Step 3: Split context edges
    print("\n[3/5] Splitting context edges...")
    random.seed(random_seed)
    context_edges_shuffled = context_edges.copy()
    random.shuffle(context_edges_shuffled)

    n_train = int(len(context_edges_shuffled) * train_ratio)
    n_valid = int(len(context_edges_shuffled) * valid_ratio)

    train_edges = context_edges_shuffled[:n_train]
    valid_edges = context_edges_shuffled[n_train : n_train + n_valid]
    test_edges = context_edges_shuffled[n_train + n_valid :]

    print(f"  - Train: {len(train_edges)} edges ({train_ratio * 100:.0f}%)")
    print(f"  - Valid: {len(valid_edges)} edges ({valid_ratio * 100:.0f}%)")
    print(f"  - Test: {len(test_edges)} edges ({test_ratio * 100:.0f}%)")

    # Step 4: Save files
    print("\n[4/5] Saving files...")
    os.makedirs(output_dir, exist_ok=True)

    export_triples(train_edges, f"{output_dir}/train.txt")
    export_triples(valid_edges, f"{output_dir}/valid.txt")
    export_triples(test_edges, f"{output_dir}/test.txt")
    export_triples(list(backbone_edges), f"{output_dir}/validation_pool.txt")

    print(f"  ✓ Saved to {output_dir}/")

    # Step 5: Validation checks
    print("\n[5/5] Running validation checks...")

    # Check for overlap between splits
    train_set = set(train_edges)
    valid_set = set(valid_edges)
    test_set = set(test_edges)
    backbone_set = set(backbone_edges)

    overlaps = []
    if train_set & valid_set:
        overlaps.append(f"train-valid: {len(train_set & valid_set)}")
    if train_set & test_set:
        overlaps.append(f"train-test: {len(train_set & test_set)}")
    if valid_set & test_set:
        overlaps.append(f"valid-test: {len(valid_set & test_set)}")
    if train_set & backbone_set:
        overlaps.append(f"train-backbone: {len(train_set & backbone_set)}")
    if valid_set & backbone_set:
        overlaps.append(f"valid-backbone: {len(valid_set & backbone_set)}")
    if test_set & backbone_set:
        overlaps.append(f"test-backbone: {len(test_set & backbone_set)}")

    if overlaps:
        print(f"  ❌ FAILED: Overlaps detected: {', '.join(overlaps)}")
        raise ValueError("Data leakage detected!")
    else:
        print(f"  ✅ PASSED: No overlaps between splits")

    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)

    return {
        "train": train_edges,
        "valid": valid_edges,
        "test": test_edges,
        "validation_pool": list(backbone_edges),
    }


def merge_and_export_subgraphs(
    params: Params,
    adj_list: List,
    entity2id: dict,
    relation2id: dict,
    id2entity: dict,
    id2relation: dict,
    output_path: str,
) -> int:
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
    subgraphs = load_subgraphs_from_db(params.db_path, split_name="query")
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
    export_triples(labeled_edges, output_path, format="tsv")

    print(f"✓ Exported {len(labeled_edges)} triples to {output_path}")
    return len(labeled_edges)


if __name__ == "__main__":
    # read the whole graph (big G) and the backbone edges (E^{q}_{b})
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(
        files
    )

    # there is a single graph object: G which combines all of the files to load
    # triplets contains namespaced set of edges from each file
    # Note: adj_list is built only from 'train' data (see data_utils.py:60)

    # Setup parameters for subgraph extraction
    params = Params(
        db_path="./lcilp/data/subgraphs_db",
        hop=3,
        enclosing_sub_graph=True,
        max_nodes_per_hop=None,
        map_size_multiplier=10,
    )

    # Create output directory for LMDB database
    os.makedirs(params.db_path, exist_ok=True)

    # Prepare graphs dict for links2subgraphs
    # For query_backbone edges, we treat them all as positive samples
    graphs = {
        "query": {
            "pos": triplets["query_backbone"],  # Backbone edges as positive
            "neg": [],  # No negative samples for now
        }
    }

    # Extract subgraphs for query_backbone edges
    print(
        f"Extracting subgraphs for {len(triplets['query_backbone'])} backbone edges..."
    )
    links2subgraphs(adj_list, graphs, params, max_label_value=None)
    print(f"Subgraphs saved to {params.db_path}")

    # Merge and export subgraphs
    output_file = "./lcilp/data/alzheimers_merged_subgraphs.txt"
    num_edges = merge_and_export_subgraphs(
        params, adj_list, entity2id, relation2id, id2entity, id2relation, output_file
    )

    # Prepare focused dataset with backbone isolation
    print("\n" + "=" * 60)
    print("Creating train/valid/test splits for HITL training")
    print("=" * 60)

    try:
        prepare_focused_dataset(
            merged_file=output_file,
            backbone_file=files["query_backbone"],
            output_dir=FOCUSED_OUTPUT_DIR,
        )
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("Dataset preparation failed. Please check input files.")
        exit(1)
