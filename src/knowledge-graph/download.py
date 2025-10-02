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
// Start with Alzheimer's and related conditions
MATCH (alzheimer)
WHERE alzheimer.id = 'MONDO:0004975'  // Alzheimer disease
OR alzheimer.name =~ '(?i).*alzheimer.*'
OR alzheimer.id IN ['MONDO:0007088', 'MONDO:0007089']  // AD type 1, type 2

// Get first-degree connections
MATCH (alzheimer)-[r1]-(connected1)
WHERE connected1.id IS NOT NULL

// Get second-degree connections to expand the network
OPTIONAL MATCH (connected1)-[r2]-(connected2)
WHERE connected2.id IS NOT NULL
AND connected2 <> alzheimer
AND (connected2.id STARTS WITH "HP:" OR
     connected2.id STARTS WITH "MONDO:" OR
     connected2.id STARTS WITH "HGNC:")

WITH COLLECT(DISTINCT alzheimer) + COLLECT(DISTINCT connected1) + COLLECT(DISTINCT connected2) as nodes,
     COLLECT(DISTINCT r1) + COLLECT(DISTINCT r2) as edges

RETURN nodes[0..1500] as nodes,
       edges[0..6000] as edges,
       SIZE(edges) as edge_count
"""

payload = json.dumps({"query": query})
headers = {"Content-Type": "application/json", "Accept": "application/json"}

response = requests.request("POST", url, headers=headers, data=payload)
response.raise_for_status()

os.makedirs(data_dir, exist_ok=True)
with open(os.path.join(data_dir, "alzheimers_subgraph.json"), "w") as f:
    f.write(response.text)
