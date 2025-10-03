#!/usr/bin/env python3
"""
Convert CSV triplet files to tab-delimited format for GNN processing.

Usage:
    python convert_csv_to_triples.py \
        --input outputs/antijoin_alzheimers_llm.csv \
        --output src/gnn/lcilp/data/alzheimers_llm_triples.txt
"""

import argparse
import os


def convert_csv_to_tsv(input_path: str, output_path: str, has_header: bool = True):
    """
    Convert CSV file to tab-delimited format.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output TSV file
        has_header: Whether the CSV has a header row to skip
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(infile):
            # Skip header if present
            if i == 0 and has_header:
                continue

            # Convert comma to tab
            line = line.strip()
            if line:
                parts = line.split(",")
                if len(parts) == 3:
                    outfile.write("\t".join(parts) + "\n")
                else:
                    print(f"⚠️  Warning: Skipping malformed line {i+1}: {line}")

    print(f"✓ Converted {input_path} → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV triplet files to tab-delimited format"
    )
    parser.add_argument(
        "--input", required=True, help="Input CSV file path (or comma-separated list)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV file path (or comma-separated list)",
    )
    parser.add_argument(
        "--no-header", action="store_true", help="CSV has no header row"
    )

    args = parser.parse_args()

    # Handle multiple files
    input_files = args.input.split(",")
    output_files = args.output.split(",")

    if len(input_files) != len(output_files):
        raise ValueError(
            f"Number of input files ({len(input_files)}) must match output files ({len(output_files)})"
        )

    for input_path, output_path in zip(input_files, output_files):
        convert_csv_to_tsv(
            input_path.strip(), output_path.strip(), has_header=not args.no_header
        )


if __name__ == "__main__":
    main()
