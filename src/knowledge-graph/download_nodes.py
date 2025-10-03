"""
Download Node Information from MonarchKG

Retrieves detailed node information (id, label, description) for nodes
in the Alzheimer's subgraph or any provided list of node IDs.
"""

import requests
import json
import csv
import os
from typing import List, Set

def extract_node_ids_from_triples(triples_file: str) -> Set[str]:
    """Extract all unique node IDs from a triples CSV file."""
    node_ids = set()
    
    with open(triples_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_ids.add(row['subject'])
            node_ids.add(row['object'])
    
    return node_ids

def get_node_data(node_ids: List[str]) -> List[dict]:
    """
    Get node information from MonarchKG for a list of node IDs in a single query.
    
    Args:
        node_ids: List of node IDs to retrieve data for
        
    Returns:
        List of node dictionaries with id, name, and description
    """
    url = "https://robokop-automat.apps.renci.org/monarch-kg/cypher"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    # Create a single Cypher query for all nodes
    node_ids_str = "', '".join(node_ids)
    query = f"""
    MATCH (n)
    WHERE n.id IN ['{node_ids_str}']
    RETURN DISTINCT
        n.id AS id,
        n.name AS name,
        n.description AS description
    ORDER BY n.id
    """
    
    payload = json.dumps({"query": query})
    
    try:
        print(f"Fetching data for {len(node_ids)} nodes in single query...")
        response = requests.post(url, headers=headers, data=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract node data from response
        all_nodes = []
        if "results" in data and len(data["results"]) > 0:
            for item in data["results"][0]["data"]:
                row = item["row"]
                node_info = {
                    "id": row[0],
                    "name": row[1] if row[1] is not None else "",
                    "description": row[2] if row[2] is not None else ""
                }
                all_nodes.append(node_info)
        
        print(f"Successfully retrieved data for {len(all_nodes)} nodes")
        return all_nodes
        
    except Exception as e:
        print(f"Error fetching node data: {e}")
        return []

def main():
    """Main function to download node data for Alzheimer's subgraph."""
    data_dir = "data"
    triples_file = os.path.join(data_dir, "alzheimers_triples.csv")
    nodes_file = os.path.join(data_dir, "alzheimers_nodes.csv")
    
    # Check if triples file exists
    if not os.path.exists(triples_file):
        print(f"Triples file not found: {triples_file}")
        print("Please run the triples download first.")
        return
    
    # Extract node IDs from triples
    print("Extracting node IDs from triples...")
    node_ids = extract_node_ids_from_triples(triples_file)
    print(f"Found {len(node_ids)} unique nodes")
    
    # Get node data
    print("Downloading node information...")
    nodes_data = get_node_data(list(node_ids))
    
    # Save both CSV and JSON formats
    os.makedirs(data_dir, exist_ok=True)
    
    # Save as CSV (for compatibility)
    with open(nodes_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'description'])
        writer.writeheader()
        writer.writerows(nodes_data)
    
    # Save as Cytoscape.js compatible JSON
    cytoscape_nodes = []
    for node in nodes_data:
        cytoscape_node = {
            "data": {
                "id": node["id"],
                "label": node["name"],
                "description": node["description"]
            }
        }
        cytoscape_nodes.append(cytoscape_node)
    
    # Create the Cytoscape.js nodes structure
    cytoscape_format = {
        "elements": {
            "nodes": cytoscape_nodes
        }
    }
    
    json_file = os.path.join(data_dir, "alzheimers_nodes.json")
    with open(json_file, 'w') as f:
        json.dump(cytoscape_format, f, indent=2)
    
    print(f"Node data saved to {nodes_file}")
    print(f"Cytoscape.js nodes saved to {json_file}")
    print(f"Retrieved data for {len(nodes_data)} nodes")

if __name__ == "__main__":
    main()