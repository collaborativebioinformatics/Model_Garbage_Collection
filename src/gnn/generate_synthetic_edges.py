#!/usr/bin/env python3
"""
Generate mock synthetic edges for testing HITL pipeline.

This script creates placeholder synthetic edges that mimic what LLM/RAG will
eventually produce. It generates three types:
- train_random.txt: Random synthetic edges for training
- valid_random.txt: Random synthetic edges for validation (human review pool)
- test_random.txt: Random synthetic edges for final evaluation

The synthetic edges are a mix of:
1. Real edges from the graph (to test true positives)
2. Corrupted edges (head or tail replaced randomly)
3. Random entity pairs

This allows testing the full HITL pipeline before real LLM/RAG integration.
"""

import sys
import os
import random
from typing import List, Tuple, Set

# Add lcilp to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lcilp"))

from utils.data_utils import process_files


def load_triplets(file_path: str) -> List[Tuple[str, str, str]]:
    """Load triplets from TSV file."""
    triplets = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    triplets.append(tuple(parts))
    return triplets


def save_triplets(triplets: List[Tuple[str, str, str]], file_path: str):
    """Save triplets to TSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for h, r, t in triplets:
            f.write(f"{h}\t{r}\t{t}\n")


def generate_synthetic_edges(
    all_edges: List[Tuple[str, str, str]],
    all_entities: List[str],
    all_relations: List[str],
    num_edges: int,
    true_positive_ratio: float = 0.3,
    corrupted_ratio: float = 0.5,
    random_ratio: float = 0.2,
    random_seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """
    Generate synthetic edges with controlled quality distribution.

    Args:
        all_edges: All real edges from the graph
        all_entities: All entity IDs
        all_relations: All relation types
        num_edges: Number of synthetic edges to generate
        true_positive_ratio: Proportion of real edges (high quality)
        corrupted_ratio: Proportion of corrupted edges (plausible but wrong)
        random_ratio: Proportion of random edges (clearly wrong)
        random_seed: For reproducibility

    Returns:
        List of synthetic edge triplets
    """
    random.seed(random_seed)
    edge_set = set(all_edges)
    synthetic_edges = []

    # Calculate counts
    num_true = int(num_edges * true_positive_ratio)
    num_corrupted = int(num_edges * corrupted_ratio)
    num_random = num_edges - num_true - num_corrupted

    print(f"  Generating {num_edges} synthetic edges:")
    print(f"    - True positives: {num_true} ({true_positive_ratio * 100:.0f}%)")
    print(f"    - Corrupted edges: {num_corrupted} ({corrupted_ratio * 100:.0f}%)")
    print(f"    - Random edges: {num_random} ({random_ratio * 100:.0f}%)")

    # 1. True positives: Sample real edges
    if num_true > 0:
        true_positive_edges = random.sample(all_edges, min(num_true, len(all_edges)))
        synthetic_edges.extend(true_positive_edges)

    # 2. Corrupted edges: Take real edge and corrupt head or tail
    if num_corrupted > 0:
        for _ in range(num_corrupted):
            h, r, t = random.choice(all_edges)

            # Randomly corrupt head or tail
            if random.random() < 0.5:
                # Corrupt head
                new_h = random.choice(all_entities)
                candidate = (new_h, r, t)
            else:
                # Corrupt tail
                new_t = random.choice(all_entities)
                candidate = (h, r, new_t)

            # Ensure we don't accidentally create a real edge
            if candidate not in edge_set:
                synthetic_edges.append(candidate)

    # 3. Random edges: Completely random entity pairs
    if num_random > 0:
        attempts = 0
        max_attempts = num_random * 10

        while len(synthetic_edges) < num_edges and attempts < max_attempts:
            h = random.choice(all_entities)
            r = random.choice(all_relations)
            t = random.choice(all_entities)

            candidate = (h, r, t)

            # Skip if this is a real edge or duplicate
            if candidate not in edge_set and candidate not in synthetic_edges:
                synthetic_edges.append(candidate)

            attempts += 1

    # Shuffle to mix edge types
    random.shuffle(synthetic_edges)

    return synthetic_edges[:num_edges]


def main():
    print("=" * 60)
    print("Generating Mock Synthetic Edges for HITL Testing")
    print("=" * 60)

    # Load all data to get entity/relation vocabulary
    print("\n[1/5] Loading graph data...")
    files = {
        "train": "./lcilp/data/alzheimers_triples.txt",
    }

    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(
        files, None
    )

    print(f"  - Entities: {len(entity2id)}")
    print(f"  - Relations: {len(relation2id)}")
    print(f"  - Total edges: {len(triplets['train'])}")

    # Convert to string format
    # Note: triplets are stored as [head_id, tail_id, relation_id]
    all_edges = [
        (id2entity[h], id2relation[r], id2entity[t]) for h, t, r in triplets["train"]
    ]
    all_entities = list(id2entity.values())
    all_relations = list(id2relation.values())

    # Load original splits to determine sizes
    print("\n[2/5] Loading original splits...")
    original_train = load_triplets("./lcilp/data/AlzheimersKG/original/train.txt")
    original_valid = load_triplets("./lcilp/data/AlzheimersKG/original/valid.txt")
    original_test = load_triplets("./lcilp/data/AlzheimersKG/original/test.txt")

    print(f"  - Original train: {len(original_train)} edges")
    print(f"  - Original valid: {len(original_valid)} edges")
    print(f"  - Original test: {len(original_test)} edges")

    # Generate synthetic splits (same size as original for now)
    # In production, synthetic would be much larger
    print("\n[3/5] Generating synthetic training edges...")
    synthetic_train = generate_synthetic_edges(
        all_edges=all_edges,
        all_entities=all_entities,
        all_relations=all_relations,
        num_edges=len(original_train),  # Same size as original train
        true_positive_ratio=0.3,
        corrupted_ratio=0.5,
        random_ratio=0.2,
        random_seed=42,
    )

    print("\n[4/5] Generating synthetic validation edges...")
    synthetic_valid = generate_synthetic_edges(
        all_edges=all_edges,
        all_entities=all_entities,
        all_relations=all_relations,
        num_edges=len(original_valid),  # Same size as original valid
        true_positive_ratio=0.3,
        corrupted_ratio=0.5,
        random_ratio=0.2,
        random_seed=43,  # Different seed
    )

    print("\n[5/5] Generating synthetic test edges...")
    synthetic_test = generate_synthetic_edges(
        all_edges=all_edges,
        all_entities=all_entities,
        all_relations=all_relations,
        num_edges=len(original_test),  # Same size as original test
        true_positive_ratio=0.3,
        corrupted_ratio=0.5,
        random_ratio=0.2,
        random_seed=44,  # Different seed
    )

    # Save synthetic edges
    print("\n[6/6] Saving synthetic edges...")
    output_dir = "./lcilp/data/AlzheimersKG/synthetic"

    save_triplets(synthetic_train, f"{output_dir}/train_random.txt")
    save_triplets(synthetic_valid, f"{output_dir}/valid_random.txt")
    save_triplets(synthetic_test, f"{output_dir}/test_random.txt")

    print(f"  ✓ Saved to {output_dir}/")
    print(f"    - train_random.txt: {len(synthetic_train)} edges")
    print(f"    - valid_random.txt: {len(synthetic_valid)} edges")
    print(f"    - test_random.txt: {len(synthetic_test)} edges")

    # Summary
    print("\n" + "=" * 60)
    print("✓ Mock Synthetic Edge Generation Complete!")
    print("=" * 60)
    print("\nData Structure:")
    print("data/AlzheimersKG/")
    print("├── original/")
    print(f"│   ├── train.txt ({len(original_train)} edges)")
    print(f"│   ├── valid.txt ({len(original_valid)} edges)")
    print(f"│   └── test.txt ({len(original_test)} edges)")
    print("└── synthetic/")
    print(f"    ├── train_random.txt ({len(synthetic_train)} edges)")
    print(f"    ├── valid_random.txt ({len(synthetic_valid)} edges)")
    print(f"    └── test_random.txt ({len(synthetic_test)} edges)")
    print("\nNext steps:")
    print("1. Update score_edges.py to score synthetic/valid_random.txt")
    print("2. Update train.py to load original + synthetic edges")
    print("3. Test end-to-end HITL workflow")


if __name__ == "__main__":
    main()
