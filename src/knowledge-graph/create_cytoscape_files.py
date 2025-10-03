"""
Create Cytoscape.js JSON files from node and edge data

This script combines node information from alzheimers_nodes.json with edge data
from the three antijoin CSV files to create Cytoscape.js compatible JSON files.

Input files:
- data/alzheimers_nodes.json (node information)
- outputs/antijoin_alzheimers_llm_rag.csv (edges)
- outputs/antijoin_alzheimers_llm.csv (edges)
- outputs/antijoin_alzheimers_random.csv (edges)
- outputs/antijoin_alzheimers_llm_rag-backbone.csv (edges)
- outputs/antijoin_alzheimers_llm-backbone.csv (edges)
- outputs/antijoin_alzheimers_random-backbone.csv (edges)

Output files:
- outputs/cytoscape/alzheimers_llm_rag.json
- outputs/cytoscape/alzheimers_llm.json
- outputs/cytoscape/alzheimers_random.json
"""

import json
import csv
import os
from typing import Dict, List, Set


def load_nodes(nodes_file: str) -> Dict[str, dict]:
    """Load node data from JSON file and create a lookup dictionary."""
    print(f"Loading node data from {nodes_file}")

    with open(nodes_file, "r") as f:
        data = json.load(f)

    # Convert nodes list to dictionary for easy lookup
    nodes_dict = {}
    for node in data["elements"]["nodes"]:
        node_id = node["data"]["id"]
        nodes_dict[node_id] = node["data"]

    print(f"Loaded {len(nodes_dict)} nodes")
    return nodes_dict


def load_edges(csv_file: str) -> List[dict]:
    """Load edge data from CSV file."""
    print(f"Loading edge data from {csv_file}")

    edges = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(
                {
                    "source": row["subject"],
                    "target": row["object"],
                    "label": row["predicate"],
                }
            )

    print(f"Loaded {len(edges)} edges")
    return edges


def get_referenced_nodes(edges: List[dict]) -> Set[str]:
    """Get all node IDs referenced in the edges (subjects and objects)."""
    referenced = set()
    for edge in edges:
        referenced.add(edge["source"])
        referenced.add(edge["target"])
    return referenced


def create_cytoscape_json(
    nodes_dict: Dict[str, dict], edges: List[dict], output_file: str
):
    """Create Cytoscape.js JSON format and save to file."""

    # Get all nodes referenced in edges
    referenced_node_ids = get_referenced_nodes(edges)
    print(f"Found {len(referenced_node_ids)} unique nodes referenced in edges")

    # Check for missing nodes and throw error if any are missing
    missing_nodes = referenced_node_ids - set(nodes_dict.keys())
    if missing_nodes:
        raise ValueError(f"Missing node data for: {sorted(missing_nodes)}")

    # Create nodes array with only referenced nodes, sorted by ID for deterministic output
    cytoscape_nodes = []
    for node_id in sorted(referenced_node_ids):
        node_data = nodes_dict[node_id]
        cytoscape_nodes.append({"data": node_data})

    # Create edges array, sorted by source, then target, then label for deterministic output
    cytoscape_edges = []
    sorted_edges = sorted(edges, key=lambda x: (x["source"], x["target"], x["label"]))
    for edge in sorted_edges:
        cytoscape_edges.append({"data": edge})

    # Create final structure
    cytoscape_json = {"elements": {"nodes": cytoscape_nodes, "edges": cytoscape_edges}}

    # Save to file with sorted keys for consistent JSON formatting
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(cytoscape_json, f, indent=2, sort_keys=True)

    print(
        f"Created {output_file} with {len(cytoscape_nodes)} nodes and {len(cytoscape_edges)} edges"
    )


def main():
    """Main function to process all three CSV files."""

    # Input files
    nodes_file = "data/alzheimers_nodes.json"
    csv_files = [
        "outputs/antijoin_alzheimers_llm_rag.csv",
        "outputs/antijoin_alzheimers_llm.csv",
        "outputs/antijoin_alzheimers_random.csv",
        "outputs/antijoin_alzheimers_llm_rag-backbone.csv",
        "outputs/antijoin_alzheimers_llm-backbone.csv",
        "outputs/antijoin_alzheimers_random-backbone.csv",
    ]

    # Check if nodes file exists
    if not os.path.exists(nodes_file):
        raise FileNotFoundError(f"Node data file not found: {nodes_file}")

    # Load node data once
    nodes_dict = load_nodes(nodes_file)

    # Process each CSV file
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found: {csv_file}, skipping...")
            continue

        # Extract the part after "antijoin_" and before ".csv"
        base_name = os.path.basename(csv_file)  # antijoin_alzheimers_llm_rag.csv
        name_part = base_name.replace("antijoin_", "").replace(
            ".csv", ""
        )  # alzheimers_llm_rag

        # Create output filename
        output_file = f"outputs/cytoscape/{name_part}.json"

        try:
            # Load edges
            edges = load_edges(csv_file)

            # Create Cytoscape JSON
            create_cytoscape_json(nodes_dict, edges, output_file)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            raise

    print("\nAll Cytoscape.js files created successfully!")


if __name__ == "__main__":
    main()
