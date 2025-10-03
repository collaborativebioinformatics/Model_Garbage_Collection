#!/usr/bin/env python3
"""
Balance Negative Sampling Script

This script balances positive and negative edges for GNN training by subsampling
synthetic (negative) edges to maintain a target positive-to-negative ratio.

Usage:
    python balance_negatives.py \\
        --original data/AlzheimersKG/original/train.txt \\
        --synthetic data/AlzheimersKG/synthetic/train_random.txt,data/AlzheimersKG/synthetic/train_llm.txt \\
        --output data/AlzheimersKG/synthetic/train_balanced.txt \\
        --ratio 2.0 \\
        --seed 42

Arguments:
    --original: Path to original (positive) edges file
    --synthetic: Comma-separated list of synthetic (negative) edge files
    --output: Base output path (e.g., "train_balanced.txt")
    --ratio: Negative-to-positive ratio (default: 2.0 for 1:2 ratio)
    --seed: Random seed for reproducibility (default: 42)

Output Files:
    Given --output "data/AlzheimersKG/synthetic/train_balanced.txt", creates:
    - train_balanced_neg.txt: Balanced negatives only (use with --synthetic_train_files)
    - train_balanced_combined.txt: Both positives and negatives (for reference)

File Format:
    Tab-separated triplets (head, relation, tail):
    MONDO:0100280\tbiolink:subclass_of\tMONDO:0019052

Edge Case Handling:
    - If synthetic < target: Use all synthetic edges, log warning
    - If synthetic > target: Subsample to exact target count
    - Preserve stratification: Sample proportionally from each synthetic file

Training Integration:
    After running this script, use the negatives file in training:

    python train.py -d AlzheimersKG -e iteration_0 \\
        --synthetic_train_files "train_balanced_neg.txt"

    Positives are automatically loaded from original/train.txt
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_edges(file_path: str) -> List[str]:
    """
    Load edges from a file.

    Args:
        file_path: Path to the edges file

    Returns:
        List of edge strings (including newlines)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r") as f:
        edges = f.readlines()

    # Filter out empty lines
    edges = [edge for edge in edges if edge.strip()]

    logger.info(f"Loaded {len(edges)} edges from {file_path}")
    return edges


def load_synthetic_edges(
    file_paths: List[str],
) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Load synthetic edges from multiple files and track their sources.

    Args:
        file_paths: List of paths to synthetic edge files

    Returns:
        Tuple of:
        - Combined list of all synthetic edges
        - List of (file_path, count) tuples for stratification

    Raises:
        FileNotFoundError: If any file doesn't exist
    """
    all_edges = []
    file_counts = []

    for file_path in file_paths:
        edges = load_edges(file_path)
        all_edges.extend(edges)
        file_counts.append((file_path, len(edges)))

    logger.info(
        f"Loaded {len(all_edges)} total synthetic edges from {len(file_paths)} files"
    )
    return all_edges, file_counts


def stratified_sample(
    file_paths: List[str],
    file_counts: List[Tuple[str, int]],
    target_count: int,
    seed: int,
) -> List[str]:
    """
    Sample edges proportionally from each synthetic file to maintain stratification.

    Args:
        file_paths: List of paths to synthetic edge files
        file_counts: List of (file_path, count) tuples
        target_count: Total number of edges to sample
        seed: Random seed for reproducibility

    Returns:
        List of sampled edges
    """
    random.seed(seed)

    total_available = sum(count for _, count in file_counts)
    sampled_edges = []

    logger.info(
        f"Stratified sampling {target_count} edges from {total_available} available:"
    )

    for file_path, count in file_counts:
        # Calculate proportional sample size
        proportion = count / total_available
        sample_size = int(target_count * proportion)

        # Ensure we sample at least 1 edge if the file has edges
        if count > 0 and sample_size == 0:
            sample_size = 1

        logger.info(
            f"  - {Path(file_path).name}: {count} edges -> sampling {sample_size} ({proportion * 100:.1f}%)"
        )

        # Load and sample from this file
        edges = load_edges(file_path)
        if sample_size >= len(edges):
            # Use all edges if sample size >= available
            sampled_edges.extend(edges)
        else:
            # Random sample
            sampled = random.sample(edges, sample_size)
            sampled_edges.extend(sampled)

    # Handle rounding errors: if we're short, add more edges randomly
    if len(sampled_edges) < target_count:
        shortage = target_count - len(sampled_edges)
        logger.info(
            f"  - Adjusting for rounding: adding {shortage} more edges randomly"
        )

        # Load all edges again and sample the shortage
        all_edges = []
        for file_path, _ in file_counts:
            all_edges.extend(load_edges(file_path))

        # Remove already sampled edges to avoid duplicates
        remaining = [e for e in all_edges if e not in sampled_edges]
        additional = random.sample(remaining, min(shortage, len(remaining)))
        sampled_edges.extend(additional)

    # Handle excess: if we're over, trim randomly
    elif len(sampled_edges) > target_count:
        excess = len(sampled_edges) - target_count
        logger.info(f"  - Adjusting for rounding: removing {excess} edges randomly")
        sampled_edges = random.sample(sampled_edges, target_count)

    return sampled_edges


def balance_negatives(
    original_path: str,
    synthetic_paths: List[str],
    output_path: str,
    ratio: float,
    seed: int,
) -> None:
    """
    Balance positive and negative edges by subsampling negatives.

    Args:
        original_path: Path to original (positive) edges file
        synthetic_paths: List of paths to synthetic (negative) edge files
        output_path: Output path for balanced training file
        ratio: Negative-to-positive ratio (e.g., 2.0 for 1:2 ratio)
        seed: Random seed for reproducibility
    """
    logger.info("=" * 70)
    logger.info("NEGATIVE SAMPLING BALANCE SCRIPT")
    logger.info("=" * 70)
    logger.info(f"Original edges: {original_path}")
    logger.info(f"Synthetic edges: {', '.join(synthetic_paths)}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Target ratio (neg:pos): {ratio}:1")
    logger.info(f"Random seed: {seed}")
    logger.info("=" * 70)

    # Load original (positive) edges
    logger.info("\n[1/4] Loading original edges...")
    original_edges = load_edges(original_path)
    num_positives = len(original_edges)

    # Calculate target negative count
    target_negatives = int(num_positives * ratio)
    logger.info(f"\nTarget counts:")
    logger.info(f"  - Positives: {num_positives}")
    logger.info(f"  - Negatives: {target_negatives} ({ratio}:1 ratio)")

    # Load synthetic (negative) edges
    logger.info(f"\n[2/4] Loading synthetic edges...")
    all_synthetic_edges, file_counts = load_synthetic_edges(synthetic_paths)
    num_available_negatives = len(all_synthetic_edges)

    # Check if we have enough negatives
    logger.info(f"\n[3/4] Balancing negatives...")
    if num_available_negatives < target_negatives:
        logger.warning(
            f"WARNING: Not enough synthetic edges! "
            f"Available: {num_available_negatives}, Target: {target_negatives}"
        )
        logger.warning(f"Using ALL {num_available_negatives} available synthetic edges")
        sampled_negatives = all_synthetic_edges
    elif num_available_negatives == target_negatives:
        logger.info(
            f"Perfect match! Using all {num_available_negatives} synthetic edges"
        )
        sampled_negatives = all_synthetic_edges
    else:
        logger.info(
            f"Subsampling {target_negatives} from {num_available_negatives} synthetic edges"
        )
        sampled_negatives = stratified_sample(
            synthetic_paths, file_counts, target_negatives, seed
        )

    # Write output files (separate positive and negative files)
    logger.info(f"\n[4/4] Writing balanced training files...")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Determine output file names
    # If output is "train_balanced.txt", create:
    #   - train_balanced_neg.txt (negatives only, for use with --synthetic_train_files)
    #   - train_balanced_combined.txt (both, for reference)
    output_stem = output_path_obj.stem  # e.g., "train_balanced"
    output_dir = output_path_obj.parent

    neg_output_path = output_dir / f"{output_stem}_neg.txt"
    combined_output_path = output_dir / f"{output_stem}_combined.txt"

    # Write negatives file (this is what train.py will use)
    with open(neg_output_path, "w") as f:
        for edge in sampled_negatives:
            f.write(edge)
    logger.info(f"  ✓ Wrote negatives to: {neg_output_path}")

    # Write combined file (for reference/debugging)
    with open(combined_output_path, "w") as f:
        # Write original edges first
        for edge in original_edges:
            f.write(edge)
        # Write sampled synthetic edges
        for edge in sampled_negatives:
            f.write(edge)
    logger.info(f"  ✓ Wrote combined file to: {combined_output_path}")

    total_edges = num_positives + len(sampled_negatives)
    actual_ratio = len(sampled_negatives) / num_positives if num_positives > 0 else 0

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Negatives file: {neg_output_path}")
    logger.info(f"Combined file: {combined_output_path}")
    logger.info(f"Total edges (combined): {total_edges}")
    logger.info(f"  - Positives (original): {num_positives}")
    logger.info(f"  - Negatives (sampled): {len(sampled_negatives)}")
    logger.info(f"Actual ratio (neg:pos): {actual_ratio:.2f}:1")
    logger.info(f"Target ratio (neg:pos): {ratio}:1")

    if abs(actual_ratio - ratio) > 0.01:
        logger.warning(f"WARNING: Actual ratio differs from target!")
    else:
        logger.info("SUCCESS: Target ratio achieved!")

    logger.info("=" * 70)
    logger.info("\nUSAGE IN TRAINING:")
    logger.info(f"  python train.py -d <DATASET> -e <EXP> \\")
    logger.info(f'    --synthetic_train_files "{neg_output_path.name}"')
    logger.info("=" * 70)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Balance positive and negative edges for GNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Balance with 1:2 ratio (default)
    python balance_negatives.py \\
        --original data/AlzheimersKG/original/train.txt \\
        --synthetic data/AlzheimersKG/synthetic/train_random.txt \\
        --output data/AlzheimersKG/synthetic/train_balanced.txt

    # Creates: train_balanced_neg.txt, train_balanced_combined.txt
    # Use in training: --synthetic_train_files "train_balanced_neg.txt"

    # Balance with 1:3 ratio (multiple synthetic files)
    python balance_negatives.py \\
        --original data/AlzheimersKG/original/train.txt \\
        --synthetic data/AlzheimersKG/synthetic/train_random.txt,data/AlzheimersKG/synthetic/train_llm.txt \\
        --output data/AlzheimersKG/synthetic/train_balanced.txt \\
        --ratio 3.0

    # Balance with custom seed
    python balance_negatives.py \\
        --original data/AlzheimersKG/original/train.txt \\
        --synthetic data/AlzheimersKG/synthetic/train_random.txt \\
        --output data/AlzheimersKG/synthetic/train_balanced.txt \\
        --seed 12345
        """,
    )

    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original (positive) edges file",
    )

    parser.add_argument(
        "--synthetic",
        type=str,
        required=True,
        help="Comma-separated list of synthetic (negative) edge files",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for balanced training file",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=2.0,
        help="Negative-to-positive ratio (default: 2.0 for 1:2 ratio)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Parse synthetic file paths
    synthetic_paths = [path.strip() for path in args.synthetic.split(",")]

    try:
        balance_negatives(
            original_path=args.original,
            synthetic_paths=synthetic_paths,
            output_path=args.output,
            ratio=args.ratio,
            seed=args.seed,
        )
    except Exception as e:
        logger.error(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
