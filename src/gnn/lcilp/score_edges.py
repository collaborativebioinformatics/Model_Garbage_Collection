#!/usr/bin/env python3
"""
Score validation edges (original + synthetic) using trained GNN model.

This script is part of the HITL (Human-in-the-Loop) pipeline. It scores
ALL validation edges (both original Monarch KG and synthetic LLM-generated)
to identify uncertain edges for human review.

Key points:
- Original edges: Trusted but not 100% perfect - can contain errors
- Synthetic edges: AI-generated, more likely to contain errors
- Both should be reviewed by domain experts

Usage:
    # Score both original and synthetic validation edges
    python score_edges.py \
        --experiment_name hitl_iter0 \
        --dataset AlzheimersKG \
        --validation_files original/valid.txt,synthetic/valid_random.txt \
        --output ../../data/hitl/iteration_1/validation_scores.jsonl

    # Subsequent iterations only score unreviewed edges
    python score_edges.py \
        --experiment_name hitl_iter1 \
        --dataset AlzheimersKG \
        --validation_files original/valid.txt,synthetic/valid_random.txt \
        --exclude_reviewed ../../data/hitl/iteration_1/edges_to_review.jsonl \
        --output ../../data/hitl/iteration_2/validation_scores.jsonl
"""

import os
import sys
import argparse
import logging
import json
import torch
from pathlib import Path
from scipy.sparse import SparseEfficiencyWarning
from warnings import simplefilter

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
from torch.utils.data import DataLoader


def load_triplets(file_path):
    """Load triplets from TSV file."""
    triplets = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    triplets.append(parts)
    return triplets


def score_edges(params, graph_classifier, dataset):
    """
    Score edges using the trained model.

    Returns:
        List of (triplet, score) tuples
    """
    scores = []
    triplets = []

    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        collate_fn=params.collate_fn,
    )

    graph_classifier.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            data_pos, targets_pos, data_neg, targets_neg = params.move_batch_to_device(
                batch, params.device
            )

            # Only score positive samples (the validation pool edges)
            score_pos = graph_classifier(data_pos)
            scores += score_pos.squeeze(1).detach().cpu().tolist()

    return scores


def export_scores(triplets, scores, output_file):
    """Export scored edges to JSONL format."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for triplet, score in zip(triplets, scores):
            record = {
                "edge_id": f"{triplet[0]}_{triplet[1]}_{triplet[2]}",
                "triplet": triplet,
                "score": float(score),
            }
            f.write(json.dumps(record) + "\n")

    logging.info(f"✓ Exported {len(scores)} scored edges to {output_file}")


def main(params):
    simplefilter(action="ignore", category=UserWarning)
    simplefilter(action="ignore", category=SparseEfficiencyWarning)

    # Parse validation files to score (both original and synthetic)
    edge_files = [f.strip() for f in params.validation_files.split(",")]
    print("=" * 60)
    print("Scoring Validation Edges (Original + Synthetic)")
    print("=" * 60)
    print(f"Files to score: {', '.join(edge_files)}")
    print(
        "Note: Both original (trusted) and synthetic (AI-generated) edges are scored."
    )

    # Load trained model
    print(f"\n[1/4] Loading model from {params.experiment_name}...")
    graph_classifier = initialize_model(params, None, load_model=True)
    print(f"  ✓ Model loaded from experiments/{params.experiment_name}/")

    # Build file_paths dict for graph construction
    # Note: 'train' is required by process_files() to build adjacency list
    params.file_paths = {
        "train": os.path.join(
            params.main_dir, f"data/{params.dataset}/original/train.txt"
        ),
    }
    # Add all validation files as separate splits
    for i, edge_file in enumerate(edge_files):
        split_name = f"validation_{i}"
        params.file_paths[split_name] = os.path.join(
            params.main_dir, f"data/{params.dataset}/{edge_file}"
        )

    splits_to_score = [f"validation_{i}" for i in range(len(edge_files))]
    db_path_suffix = "validation"

    # Generate subgraphs
    print(f"\n[2/4] Extracting subgraphs for edges...")
    params.db_path = os.path.join(
        params.main_dir,
        f"data/{params.dataset}/{db_path_suffix}_subgraphs_{params.experiment_name}",
    )

    generate_subgraph_datasets(
        params,
        splits=splits_to_score,
        saved_relation2id=graph_classifier.relation2id,
        max_label_value=graph_classifier.gnn.max_label_value,
    )
    print(f"  ✓ Subgraphs extracted to {params.db_path}")

    # Score all edge files
    print(f"\n[3/4] Loading and scoring datasets...")
    all_triplets = []
    all_scores = []

    for i, split_name in enumerate(splits_to_score):
        # Determine original file for loading triplets
        edge_file = edge_files[i]
        triplet_file = os.path.join(
            params.main_dir, f"data/{params.dataset}/{edge_file}"
        )

        print(f"  Processing {split_name}...")

        # Load dataset
        dataset = SubgraphDataset(
            params.db_path,
            f"{split_name}_pos",
            f"{split_name}_neg",  # Will be empty
            params.file_paths,
            graph_classifier.relation2id,
            add_traspose_rels=params.add_traspose_rels,
            num_neg_samples_per_link=0,  # No negative samples
            use_kge_embeddings=params.use_kge_embeddings,
            dataset=params.dataset,
            kge_model=params.kge_model,
            file_name=split_name,
        )

        # Score edges
        scores = score_edges(params, graph_classifier, dataset)

        # Load original triplets
        triplets = load_triplets(triplet_file)

        all_triplets.extend(triplets)
        all_scores.extend(scores)

        print(f"    ✓ Scored {len(scores)} edges from {split_name}")

    # Export results
    print(f"\n[4/4] Exporting results...")
    export_scores(all_triplets, all_scores, params.output)

    # Summary
    print("\n" + "=" * 60)
    print("Scoring Complete")
    print("=" * 60)
    print(f"Total edges scored: {len(all_scores)}")
    print(f"\nScore distribution:")
    print(f"  Min:  {min(all_scores):.3f}")
    print(f"  Mean: {sum(all_scores) / len(all_scores):.3f}")
    print(f"  Max:  {max(all_scores):.3f}")
    print(f"\nResults saved to: {params.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Score validation pool edges")

    # Experiment params
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        required=True,
        help="Name of trained model experiment",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name (e.g., AlzheimersKG)",
    )

    # Validation files to score (both original and synthetic)
    parser.add_argument(
        "--validation_files",
        type=str,
        required=True,
        help="Comma-separated validation files (original + synthetic), e.g., 'original/valid.txt,synthetic/valid_random.txt'",
    )
    parser.add_argument(
        "--exclude_reviewed",
        type=str,
        default=None,
        help="Path to edges_to_review.jsonl from previous iteration to exclude already-reviewed edges",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for scores JSONL file"
    )

    # Model params (should match training)
    parser.add_argument(
        "--hop", type=int, default=3, help="Enclosing subgraph hop number"
    )
    parser.add_argument(
        "--max_nodes_per_hop",
        "-max_h",
        type=int,
        default=None,
        help="if > 0, upper bound the # nodes per hop by subsampling",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="Which GPU to use (-1 for CPU)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for scoring"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloading processes"
    )

    # Graph construction params
    parser.add_argument(
        "--max_links",
        type=int,
        default=1000000,
        help="Maximum number of links to process",
    )
    parser.add_argument(
        "--enclosing_sub_graph",
        "-en",
        type=bool,
        default=True,
        help="Whether to use enclosing subgraph",
    )
    parser.add_argument(
        "--map_size_multiplier",
        type=int,
        default=10,
        help="LMDB map size multiplier for subgraph database",
    )
    parser.add_argument(
        "--add_traspose_rels",
        "-tr",
        type=bool,
        default=False,
        help="Whether to append adj matrix list with symmetric relations",
    )
    parser.add_argument(
        "--num_neg_samples_per_link",
        "-neg",
        type=int,
        default=1,
        help="Number of negative samples per positive link (not used for scoring)",
    )
    parser.add_argument(
        "--use_kge_embeddings",
        "-kge",
        type=bool,
        default=False,
        help="Whether to use pretrained KGE embeddings",
    )
    parser.add_argument(
        "--kge_model", type=str, default="TransE", help="KGE model to use"
    )

    # Constraint params
    parser.add_argument(
        "--constrained_neg_prob",
        "-cn",
        type=float,
        default=0.0,
        help="Probability of negative samples being constrained",
    )

    args = parser.parse_args()

    # Set up paths
    args.main_dir = os.path.join(os.path.dirname(__file__))
    args.exp_dir = os.path.join(args.main_dir, "experiments", args.experiment_name)

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")

    # Set collate function
    args.collate_fn = collate_dgl
    args.move_batch_to_device = move_batch_to_device_dgl

    # File paths
    args.file_paths = {
        "train": os.path.join(args.main_dir, f"data/{args.dataset}/train.txt"),
        "valid": os.path.join(args.main_dir, f"data/{args.dataset}/valid.txt"),
    }

    main(args)
