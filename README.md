# KG Model Garbage Collection

![KG-LLM/Model Garbage Collection](logo.svg)

## Project Overview
Biomedical knowledge graphs are powerful tools for linking genes, diseases, and phenotypes — but when AI models generate new edges, they often hallucinate or introduce errors. Our project focuses on pruning these errors. The KG Model Garbage Collection Tool is a proof-of-concept for how a combination of human review, grounded AI, and graph learning can work together to keep biomedical knowledge graphs accurate and trustworthy.

The KG Model Garbage Collection Tool uses a subset of a trusted graph (Monarch) and randomly removes some edges. Then, three strategies are used to fill the missing edges: random guessing, a general LLM, and an LLM using biomedical RAG. Participants (SMEs) validate (some of) these edges through a simple interface to evaluate how close each method comes to the truth. This data is used to train a graph neural network to see if it can automatically spot questionable edges and flag them for review and removal. The resulting knowledge graph is tested against the original, trusted knowledge graph. 

## Problem
Generating knowledge graphs at scale requires the use of Large Language Models (LLMs), which can introduce errors and hallucinations. In biomedicine, mistakes could lead to errors that cause harm to people. While Human-in-the-Loop (HITL) approaches can mitigate risks, they are not scalable solutions for large, complex knowledge graphs.

## Solution
The KG Model Garbage Collection Tool is a proof-of-concept (PoC) allowing curators to probe a KG using real scientific questions, provide feedback, and use that feedback to train a GNN. This tool is a (proof-of-concept) PoC to expand the impact of human curation.

<img width="561" height="518" alt="Screenshot 2025-10-03 at 9 23 05 AM" src="https://github.com/user-attachments/assets/933c615e-d665-4841-a5d6-1bd89adf72d0" />


## Quickstart Instructions
### Prerequisites

### Order of execution
1. **src/knowledge-graph/download.py** - download a subgraph from Monarch KG (and node data including id, label, & description)
2. **src/knowledge-graph/triples_to_csv.py** - convert the downloaded triples from JSON to CSV file
3. **Edge_Assignore.ipynb** - randomly remove some edges from the downloaded triples and use 3 strategies to rebuild the edges (random, LLM, LLM-RAG)
4. **src/knowledge-graph/extract.py** - extract the "backbone" of the graph for input to GNN
5. **src/knowledge-graph/create_cytoscape_files.py** - create files for visualization in Cytoscape with node & edge data for each rebuilt knowledge graph & associated backbones

## Project Goals
1. **Quality Assurance**: Develop automated methods to identify potentially erroneous edges in biomedical knowledge graphs
2. **Human-in-the-Loop Validation**: Create dynamic, scalable frameworks for expert validation of AI-generated graph content
3. **Comparative Analysis**: Evaluate different edge prediction strategies (random, LLM-based, RAG-enhanced) against ground truth
4. **Graph Neural Network Training**: Train models to automatically detect questionable edges using human feedback
5. **Validation Framework**: Establish robust methods for comparing reconstructed graphs against trusted knowledge bases

## Data Sources
The KG Model Garbage Collection tool uses and displays data and algorithms from the Monarch Initiative. The Monarch Initiative (https://monarchinitiative.org) makes biomedical knowledge exploration more efficient and effective by providing tools for genotype-phenotype analysis, genomic diagnostics, and precision medicine across broad areas of disease.
