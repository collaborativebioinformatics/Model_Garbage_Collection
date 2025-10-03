#!/usr/bin/env python3
"""
Score all edges in a graph using trained GNN model (post-HITL inference).

This script is for DEPLOYMENT/INFERENCE after HITL training completes.
It can score ANY set of edges in ANY graph, with no restrictions on:
- Edge source (original KG, synthetic, or any other source)
- Graph structure (can score edges from different graphs)
- Validation/training splits (this is pure inference mode)

Key differences from score_edges.py:
- No validation file restrictions (score ANY edges)
- No exclude_reviewed logic (this is post-HITL)
- Optimized for large-scale batch processing
- Clearer focus on inference/deployment use case

Usage:
    # Score edges from a file
    python score_all_edges.py \
        --experiment_name hitl_final \
        --dataset AlzheimersKG \
        --edge_file /path/to/edges_to_score.txt \
        --output /path/to/scored_edges.jsonl \
        --hop 3

    # Score with custom batch size for memory optimization
    python score_all_edges.py \
        --experiment_name hitl_final \
        --dataset AlzheimersKG \
        --edge_file /path/to/large_edge_set.txt \
        --output /path/to/scores.jsonl \
        --batch_size 32 \
        --hop 3

    # Filter low-confidence edges
    python score_all_edges.py \
        --experiment_name hitl_final \
        --dataset AlzheimersKG \
        --edge_file /path/to/edges.txt \
        --output /path/to/high_confidence_edges.jsonl \
        --score_threshold 0.7 \
        --hop 3

Input Format:
    Edge file should contain triplets in TSV format (one per line):
    <head_entity>\t<relation>\t<tail_entity>

Output Format:
    JSONL file with one JSON object per line:
    {"head": "...", "relation": "...", "tail": "...", "score": 0.XXX}
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
    """
    Load triplets from TSV file.

    Args:
        file_path: Path to triplet file (head\trel\ttail format)

    Returns:
        List of [head, relation, tail] triplets
    """
    triplets = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    triplets.append(parts)
                else:
                    logging.warning(
                        f"Skipping malformed line {line_num}: expected 3 columns, got {len(parts)}"
                    )
    return triplets


def score_edges_batch(params, graph_classifier, dataset):
    """
    Score all edges in the dataset using the trained model.

    Args:
        params: Parameters object with device, batch_size, etc.
        graph_classifier: Trained GNN model
        dataset: SubgraphDataset containing edges to score

    Returns:
        List of scores (floats) corresponding to each edge in dataset
    """
    scores = []

    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        collate_fn=params.collate_fn,
    )

    graph_classifier.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch (positive samples + negative samples)
            data_pos, targets_pos, data_neg, targets_neg = params.move_batch_to_device(
                batch, params.device
            )

            # Score positive samples (the edges we want to evaluate)
            score_pos = graph_classifier(data_pos)
            batch_scores = score_pos.squeeze(1).detach().cpu().tolist()
            scores.extend(batch_scores)

            if (batch_idx + 1) % 50 == 0:
                logging.info(
                    f"  Processed {(batch_idx + 1) * params.batch_size} edges..."
                )

    return scores


def export_scores_jsonl(triplets, scores, output_file, score_threshold=None):
    """
    Export scored edges to JSONL format.

    Args:
        triplets: List of [head, relation, tail] triplets
        scores: List of scores (floats)
        output_file: Path to output JSONL file
        score_threshold: Optional minimum score threshold (filter below)
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    filtered_count = 0
    total_count = len(scores)

    with open(output_file, "w") as f:
        for triplet, score in zip(triplets, scores):
            # Apply score threshold if specified
            if score_threshold is not None and score < score_threshold:
                filtered_count += 1
                continue

            record = {
                "head": triplet[0],
                "relation": triplet[1],
                "tail": triplet[2],
                "score": float(score),
            }
            f.write(json.dumps(record) + "\n")

    kept_count = total_count - filtered_count
    logging.info(f"✓ Exported {kept_count}/{total_count} scored edges to {output_file}")

    if score_threshold is not None:
        logging.info(
            f"  Filtered out {filtered_count} edges below threshold {score_threshold:.3f}"
        )


def main(params):
    simplefilter(action="ignore", category=UserWarning)
    simplefilter(action="ignore", category=SparseEfficiencyWarning)

    print("=" * 70)
    print("GNN Post-HITL Inference: Score All Edges")
    print("=" * 70)
    print(f"Model: {params.experiment_name}")
    print(f"Dataset: {params.dataset}")
    print(f"Edge file: {params.edge_file}")
    print(f"Output: {params.output}")
    if params.score_threshold is not None:
        print(f"Score threshold: {params.score_threshold:.3f}")
    print("=" * 70)

    # Step 1: Load trained model
    print(f"\n[1/4] Loading trained model...")
    graph_classifier = initialize_model(params, None, load_model=True)
    print(f"  ✓ Model loaded from experiments/{params.experiment_name}/")

    # Step 2: Load edges to score
    print(f"\n[2/4] Loading edges to score...")
    if not os.path.exists(params.edge_file):
        raise FileNotFoundError(f"Edge file not found: {params.edge_file}")

    triplets = load_triplets(params.edge_file)
    print(f"  ✓ Loaded {len(triplets)} edges from {params.edge_file}")

    # Step 3: Build subgraphs for edges
    print(f"\n[3/4] Extracting subgraphs...")

    # Build file_paths dict for graph construction
    # Note: 'train' is required by process_files() to build adjacency list
    params.file_paths = {
        "train": os.path.join(
            params.main_dir, f"data/{params.dataset}/original/train.txt"
        ),
        "inference": params.edge_file,  # Edges to score
    }

    # Generate subgraphs for inference edges
    params.db_path = os.path.join(
        params.main_dir,
        f"data/{params.dataset}/inference_subgraphs_{params.experiment_name}",
    )

    generate_subgraph_datasets(
        params,
        splits=["inference"],
        saved_relation2id=graph_classifier.relation2id,
        max_label_value=graph_classifier.gnn.max_label_value,
    )
    print(f"  ✓ Subgraphs extracted to {params.db_path}")

    # Step 4: Load dataset and score edges
    print(f"\n[4/4] Scoring edges...")

    dataset = SubgraphDataset(
        params.db_path,
        "inference_pos",
        "inference_neg",  # Will be empty (num_neg_samples_per_link=0)
        params.file_paths,
        included_relations=graph_classifier.relation2id,
        add_traspose_rels=params.add_traspose_rels,
        num_neg_samples_per_link=0,  # No negative samples for inference
        use_kge_embeddings=params.use_kge_embeddings,
        dataset=params.dataset,
        kge_model=params.kge_model,
        file_name="inference",
    )

    scores = score_edges_batch(params, graph_classifier, dataset)
    print(f"  ✓ Scored {len(scores)} edges")

    # Step 5: Export results
    print(f"\n[5/5] Exporting results...")
    export_scores_jsonl(triplets, scores, params.output, params.score_threshold)

    # Summary statistics
    print("\n" + "=" * 70)
    print("Inference Complete")
    print("=" * 70)
    print(f"Total edges scored: {len(scores)}")
    print(f"\nScore distribution:")
    print(f"  Min:  {min(scores):.4f}")
    print(f"  Mean: {sum(scores) / len(scores):.4f}")
    print(f"  Max:  {max(scores):.4f}")

    # Quantiles
    sorted_scores = sorted(scores)
    print(f"\nQuantiles:")
    print(f"  25%:  {sorted_scores[len(scores) // 4]:.4f}")
    print(f"  50%:  {sorted_scores[len(scores) // 2]:.4f}")
    print(f"  75%:  {sorted_scores[3 * len(scores) // 4]:.4f}")

    print(f"\nResults saved to: {params.output}")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Score all edges using trained GNN model (post-HITL inference)"
    )

    # Required arguments
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        required=True,
        help="Name of trained model experiment (directory in experiments/)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name (e.g., AlzheimersKG) - used for loading graph structure",
    )
    parser.add_argument(
        "--edge_file",
        type=str,
        required=True,
        help="Path to edge file in TSV format (head\\trelation\\ttail)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for scored edges JSONL file",
    )

    # Model hyperparameters (should match training configuration)
    parser.add_argument(
        "--hop",
        type=int,
        default=3,
        help="Enclosing subgraph hop number (must match training)",
    )
    parser.add_argument(
        "--max_nodes_per_hop",
        "-max_h",
        type=int,
        default=None,
        help="Upper bound on nodes per hop via subsampling (must match training)",
    )

    # Inference parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference (can differ from training)",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Optional minimum score threshold (only export edges >= threshold)",
    )

    # Compute resources
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloading processes",
    )

    # Graph construction parameters (should match training)
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
        help="Whether to use enclosing subgraph (must match training)",
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
        help="Whether to append transpose relations (must match training)",
    )
    parser.add_argument(
        "--num_neg_samples_per_link",
        "-neg",
        type=int,
        default=1,
        help="Negative samples per link (not used in inference, but required by data utils)",
    )
    parser.add_argument(
        "--use_kge_embeddings",
        "-kge",
        type=bool,
        default=False,
        help="Whether to use pretrained KGE embeddings (must match training)",
    )
    parser.add_argument(
        "--kge_model",
        type=str,
        default="TransE",
        help="KGE model to use (if use_kge_embeddings=True)",
    )
    parser.add_argument(
        "--constrained_neg_prob",
        "-cn",
        type=float,
        default=0.0,
        help="Probability of constrained negative sampling (not used in inference)",
    )

    args = parser.parse_args()

    # Set up paths
    args.main_dir = os.path.join(os.path.dirname(__file__))
    args.exp_dir = os.path.join(args.main_dir, "experiments", args.experiment_name)

    # Validate experiment directory exists
    if not os.path.exists(args.exp_dir):
        raise FileNotFoundError(
            f"Experiment directory not found: {args.exp_dir}\n"
            f"Please ensure the model was trained with experiment name '{args.experiment_name}'"
        )

    # Validate model checkpoint exists
    model_path = os.path.join(args.exp_dir, "best_graph_classifier.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"Please ensure training completed successfully for experiment '{args.experiment_name}'"
        )

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")

    # Set collate functions
    args.collate_fn = collate_dgl
    args.move_batch_to_device = move_batch_to_device_dgl

    main(args)
