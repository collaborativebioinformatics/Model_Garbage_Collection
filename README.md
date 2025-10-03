# Model Garbage Collection

![KG-LLM/Model Garbage Collection](logo.svg)

Biomedical knowledge graphs are powerful tools for linking genes, diseases, and phenotypes — but when AI models generate new edges, they often hallucinate or introduce errors. Our project focuses on pruning these errors. The Model Garbage Collection Tool is a proof-of-concept for how a combination of human review, grounded AI, and graph learning can work together to keep biomedical knowledge graphs accurate and trustworthy.

The Model Garbage Collection Tool uses a subset of a trusted graph (Monarch) and randomly removes some edges.  Then, three strategies are used to fill the missing edges: random guessing, a general LLM, and an LLM using RAG. Participants (SMEs) validate (some of) these edges through a simple interface to evaluate how close each method comes to the truth. This data is used to train a graph neural network to see if it can automatically spot questionable edges and flag them for review and removal. The resulting knowledge graph is tested it against the original, trusted knowledge graph. 

## Problem
Generating KGs at scale requires use of LLMs which can introduce errors. In biomedicine, mistakes could lead to errors that cause harm to people. One way to mitigate the risk of harm is HITL, but that is not a scalable solution for a large, complex KGs. 

## Solution
The Model Garbage Collection Tool is a PoC allowing curators to probe a KG using real scientific questions, provide feedback, and use that feedback to train a GNN. This tool is a PoC to expand the impact of human curation and remove bad data at scale.

<img width="792" height="399" alt="Screenshot 2025-10-02 at 10 54 14 AM" src="https://github.com/user-attachments/assets/b4793693-0d6c-4d3a-9e5d-cb699410ac00" />

<img width="4000" height="1848" alt="20251002_110612" src="https://github.com/user-attachments/assets/72498ccc-b6a9-438c-8483-753fcc324fcc" />


## Data Sources
The Model Garbage Collection tool uses and displays data and algorithms from the Monarch Initiative. The Monarch Initiative (https://monarchinitiative.org) makes biomedical knowledge exploration more efficient and effective by providing tools for genotype-phenotype analysis, genomic diagnostics, and precision medicine across broad areas of disease.
