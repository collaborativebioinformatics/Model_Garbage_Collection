# Model_Garbage_Collection

Biomedical knowledge graphs are powerful tools for linking genes, diseases, and phenotypes — but when AI models generate new edges, they often hallucinate or introduce errors. Our project focuses on pruning these errors. (Cool tool name) is a proof-of-concept for how a combination of human review, grounded AI, and graph learning can work together to keep biomedical knowledge graphs accurate and trustworthy.


(Cool tool name) uses a subset of a trusted graph (Monarch) and randomly removes some edges.  Then, three strategies are used to fill the missing edges: random guessing, a general LLM, and an LLM using RAG. Participants (SMEs) validate (some of) these edges through a simple interface to evaluate how close each method comes to the truth. This data is used to train a graph neural network to see if it can automatically spot questionable edges and flag them for review and removal. The resulting knowledge graph is tested it against the original, trusted knowledge graph. 


## Problem
Generating KGs at scale requires use of LLMs which can introduce errors. In biomedicine, mistakes could lead to errors that cause harm to people. One way to mitigate the risk of harm is HITL, but that is not a scalable solution for a large, complex KGs. 



## Solution
(Cool named tool) is a PoC allowing curators to probe a KG using real scientific questions, provide feedback, and use that feedback to train a GNN. This tool is a PoC to expand the impact of human curation and remove bad data at scale.

<img width="792" height="399" alt="Screenshot 2025-10-02 at 10 54 14 AM" src="https://github.com/user-attachments/assets/b4793693-0d6c-4d3a-9e5d-cb699410ac00" />

### Test data
(download.py) For purposes of the hackathon, we needed a workable test set. We queried a knowledge graph to retrieve an extended network of entities associated with Alzheimer's disease, including related genes, phenotypes, and neurodegenerative conditions. The query captures first- and second-degree connections to generate a subgraph containing approximately 3,000 to 8,000 edges, which is then saved as a JSON file for further analysis. This approach enables comprehensive exploration of the molecular and phenotypic landscape surrounding Alzheimer's disease.


