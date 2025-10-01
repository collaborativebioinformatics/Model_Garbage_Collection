# Model_Garbage_Collection

Biomedical knowledge graphs are powerful tools for linking genes, diseases, and phenotypes — but when AI models generate new edges, they often hallucinate or introduce errors. Our project focuses on pruning these errors.

We’ll start by taking a subset of a trusted graph (like Monarch) and randomly removing some edges.  Then, we’ll use three strategies to fill the gaps: random guessing, a general LLM, and an LLM using RAG. Participants will validate (some of) these edges through a simple interface, and we’ll measure how close each method comes to the truth. We’ll train a quick graph neural network to see if it can automatically spot questionable edges and flag them for review and removal, testing it against the original, trusted knowledge graph. 

The outcome is a clear proof-of-concept for how a combination of human review, grounded AI, and graph learning can work together to keep biomedical knowledge graphs accurate and trustworthy.

<img height="1500" alt="excalidraw-pruning workflow/garbage collection-v2" src="https://github.com/user-attachments/assets/3a4aac0c-b027-44e4-87c7-7dc64888fd92" />
