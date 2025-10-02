"""
Download Source Knowledge Graph

Query: "What genes, phenotypes, and related conditions are associated with Alzheimer's disease?"

Returns an extended network around Alzheimer's disease:
- Core Alzheimer's disease entity
- Associated genes and variants
- Phenotypic manifestations
- Related neurodegenerative conditions

Expected: ~3K-8K edges
"""

import requests
import json
import os

url = "https://robokop-automat.apps.renci.org/monarch-kg/cypher"

data_dir = "data"

query = """
// Get Alzheimer's disease and 1st/2nd degree network with all edges between them
MATCH (alzheimer)
WHERE alzheimer.id = 'MONDO:0004975'  // Main Alzheimer disease

// Get first-degree connections
MATCH (alzheimer)-[r1]-(connected1)
WHERE connected1.id IS NOT NULL

// Get second-degree connections to expand the network
OPTIONAL MATCH (connected1)-[r2]-(connected2)
WHERE connected2.id IS NOT NULL
  AND connected2 <> alzheimer
  AND (connected2.id STARTS WITH "HP:" OR
       connected2.id STARTS WITH "MONDO:" OR  
       connected2.id STARTS WITH "HGNC:" OR
       connected2.id STARTS WITH "CHEBI:" OR
       connected2.id STARTS WITH "GO:")

// Collect all nodes in the subgraph
WITH COLLECT(DISTINCT alzheimer) + COLLECT(DISTINCT connected1) + COLLECT(DISTINCT connected2) AS all_nodes

// Find all edges between any nodes in our subgraph
UNWIND all_nodes AS node1
UNWIND all_nodes AS node2
MATCH (node1)-[edge]->(node2)
WHERE node1 <> node2
  AND node1.id IS NOT NULL 
  AND node2.id IS NOT NULL

RETURN DISTINCT 
  node1.id as subject,
  type(edge) as predicate, 
  node2.id as object
LIMIT 15000
"""

payload = json.dumps({"query": query})
headers = {"Content-Type": "application/json", "Accept": "application/json"}

response = requests.request("POST", url, headers=headers, data=payload)
response.raise_for_status()

os.makedirs(data_dir, exist_ok=True)
with open(os.path.join(data_dir, "alzheimers_subgraph.json"), "w") as f:
    f.write(response.text)
