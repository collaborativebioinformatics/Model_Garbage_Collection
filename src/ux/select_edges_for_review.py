#!/usr/bin/env python3
"""
Select edges for human review using uncertainty sampling.

Usage:
    python select_edges_for_review.py \
        --scores data/hitl/iteration_1/pool_scores.jsonl \
        --num_samples 10 \
        --output data/hitl/iteration_1/edges_to_review.jsonl
"""

import json
import argparse
from pathlib import Path


def load_scores(scores_file):
    """Load scored edges from JSONL."""
    scores = []
    with open(scores_file, "r") as f:
        for line in f:
            scores.append(json.loads(line))
    return scores


def uncertainty_sampling(scored_edges, num_samples):
    """
    Select edges with highest uncertainty (closest to 0.5).

    Uncertainty = 1 - |score - 0.5| * 2

    Score 0.5 → uncertainty 1.0 (most uncertain)
    Score 0.0 or 1.0 → uncertainty 0.0 (most certain)
    """
    # Compute uncertainty for each edge
    for edge in scored_edges:
        edge["uncertainty"] = 1 - abs(edge["score"] - 0.5) * 2

    # Sort by uncertainty (descending)
    scored_edges.sort(key=lambda x: x["uncertainty"], reverse=True)

    # Return top N
    return scored_edges[:num_samples]


def export_for_review(selected_edges, output_file):
    """Export selected edges to JSONL for validation UI."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for edge in selected_edges:
            # Add fields needed by validation UI
            review_record = {
                "edge_id": edge["edge_id"],
                "triplet": edge["triplet"],
                "model_score": edge["score"],
                "uncertainty": edge["uncertainty"],
                "status": "pending",  # For UI tracking
            }
            f.write(json.dumps(review_record) + "\n")

    print(f"✓ Selected {len(selected_edges)} edges for review")
    print(f"✓ Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Select edges for human review")
    parser.add_argument("--scores", required=True, help="Path to pool_scores.jsonl")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of edges to select (default: 10)",
    )
    parser.add_argument(
        "--output", required=True, help="Output path for edges_to_review.jsonl"
    )
    parser.add_argument(
        "--strategy",
        default="uncertainty",
        choices=["uncertainty", "random", "low_confidence"],
        help="Selection strategy",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Selecting Edges for Human Review")
    print("=" * 60)

    # Load scores
    print(f"\n[1/3] Loading scores from {args.scores}...")
    scored_edges = load_scores(args.scores)
    print(f"  - Loaded {len(scored_edges)} scored edges")

    # Select edges
    print(
        f"\n[2/3] Selecting {args.num_samples} edges using {args.strategy} strategy..."
    )

    if args.strategy == "uncertainty":
        selected = uncertainty_sampling(scored_edges, args.num_samples)
    elif args.strategy == "random":
        import random

        selected = random.sample(scored_edges, min(args.num_samples, len(scored_edges)))
    else:
        # Low confidence: select edges with lowest scores
        scored_edges.sort(key=lambda x: x["score"])
        selected = scored_edges[: args.num_samples]

    # Export
    print(f"\n[3/3] Exporting selected edges...")
    export_for_review(selected, args.output)

    # Summary
    print("\n" + "=" * 60)
    print("Selection Complete")
    print("=" * 60)
    print(f"Selected {len(selected)} edges for review")
    print(f"\nScore distribution of selected edges:")
    scores = [e["score"] for e in selected]
    print(f"  Min:  {min(scores):.3f}")
    print(f"  Mean: {sum(scores) / len(scores):.3f}")
    print(f"  Max:  {max(scores):.3f}")

    if args.strategy == "uncertainty":
        uncertainties = [e["uncertainty"] for e in selected]
        print(f"\nUncertainty distribution:")
        print(f"  Min:  {min(uncertainties):.3f}")
        print(f"  Mean: {sum(uncertainties) / len(uncertainties):.3f}")
        print(f"  Max:  {max(uncertainties):.3f}")


if __name__ == "__main__":
    main()
