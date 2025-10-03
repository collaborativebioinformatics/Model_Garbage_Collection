# KG Model Garbage Collection

![KG-LLM/Model Garbage Collection](logo.svg)

## Our Hackathon Team
* Allen Baron (University of Maryland - Baltimore)
* Yibei Chen (MIT)
* Anne Ketter (Computercraft)
* Samarpan Mohanty (University of Nebraska - Lincoln)
* Evan Molinelli (Chan Zuckerburg Initiative)
* Van Truong (University of Pennsylvania)

## Project Overview
Biomedical knowledge graphs (KGs) are powerful tools for linking genes, diseases, and phenotypes — but when AI models generate new edges, they often hallucinate or introduce errors. Our project focuses on pruning these errors. We show how combining human review, grounded AI, and graph learning can work together to keep biomedical knowledge graphs accurate and trustworthy.

The **KG Model Garbage Collection Tool** is a proof-of-concept that:
* Starts with a trusted subset of the Monarch KG (with edges randomly removed).
* Fills in missing edges using three approaches to simulate a real-world, messy knowledge graph: random assignment (control), a general LLM, and an LLM with biomedical RAG.
* Participants, for example subject-matter experts (SMEs), are invited to click and validate (some of) these edges through a simple interface to evaluate how close each method comes to the truth.
* This data is used to train a graph neural network (GNN) to see if it can automatically spot questionable edges and flag them for review and removal.
* The resulting knowledge graph is tested against the original, trusted knowledge graph. 

## Problem
Large Language Models (LLMs) are increasingly used to scale up knowledge graphs, but they introduce errors nad hallucinations. In biomedicine, these mistakes can have real-world consequences. While Human-in-the-Loop (HITL) approaches can mitigate risks, they are not scalable solutions for large, complex knowledge graphs.

## Solution
The **KG Model Garbage Collection Tool** provides a proof-of-concept (PoC) framework allowing curators to probe a KG using real scientific questions, provide feedback, and use that feedback to train a GNN. This tool extends the impact of human curation by learning from expert human validation patterns.

<img width="auto" height="auto" alt="Screenshot 2025-10-03 at 9 23 05 AM" src="https://github.com/user-attachments/assets/933c615e-d665-4841-a5d6-1bd89adf72d0" />


## Quickstart Instructions
### What can you do with KG Garbage Model Collector?
* You can collaboratively find and remove problem edges.
* Isolate part of a large knowledge graph to curate a smaller, workable data set
* Teach a GNN to find problems and curate only the problems it identifies
* Check the problems manually - iterate until agreement on what to prune

### System Requirements

- Python 3.8 or higher
- Node.js 14.x or higher (for frontend components)
- AWS CLI configured with appropriate credentials
- Access to PubMed E-utilities (for RAG functionality)

### Dependencies

Key python dependencies include: pandas, numpy, boto3 (AWS SDK), sentence-transformers, chromadb, langchain, requests

### AWS Configuration

Configure AWS credentials for Bedrock access:

```bash
aws configure
```

Ensure access to the following AWS services:
- Amazon Bedrock (for LLM inference)
- Appropriate IAM permissions for model access

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/collaborativebioinformatics/Model_Garbage_Collection.git
cd Model_Garbage_Collection
```

2. Install dependencies:
```bash
pip install .
```

3. Configure environment variables:
```bash
export AWS_REGION=your-region
export AWS_PROFILE=your-profile
```

### App Frontend Setup
See [app/frontend/README.md](app/frontend/README.md).


### Directory Structure
Here's an overview of our filetree in this repo. 

````
Model_Garbage_Collection/
├── app/
│   └── frontend/                    # React frontend application
├── data/                            # Input datasets
│   ├── alzheimers_nodes.json
│   └── alzheimers_triples.csv
├── notebooks/                       # Jupyter notebooks for analysis
│   └── model_testing.ipynb
├── outputs/                         # Generated results and datasets
│   └── cytoscape/                   # Graph visualization files
├── src/                             # Core source code
│   ├── gnn/                         # Graph Neural Network components
│   │   ├── lcilp/                   # Link prediction implementation
│   │   │   ├── data/                # Training datasets
│   │   │   ├── ensembling/          # Model ensemble methods
│   │   │   │   ├── blend.py
│   │   │   │   ├── compute_auc.py
│   │   │   │   └── score_triplets_kge.py
│   │   │   ├── kge/                 # Knowledge graph embeddings
│   │   │   │   ├── dataloader.py
│   │   │   │   ├── model.py
│   │   │   │   └── run.py
│   │   │   ├── managers/            # Training and evaluation
│   │   │   │   ├── evaluator.py
│   │   │   │   └── trainer.py
│   │   │   ├── model/               # Neural network architectures
│   │   │   │   └── dgl/
│   │   │   │       ├── aggregators.py
│   │   │   │       ├── graph_classifier.py
│   │   │   │       ├── layers.py
│   │   │   │       └── rgcn_model.py
│   │   │   ├── subgraph_extraction/ # Graph sampling
│   │   │   │   ├── datasets.py
│   │   │   │   ├── graph_sampler.py
│   │   │   │   └── multicom.py
│   │   │   ├── utils/               # Utility functions
│   │   │   ├── graph_sampler.py
│   │   │   ├── score_edges.py
│   │   │   ├── train.py
│   │   │   └── test_*.py
│   │   ├── extract.py
│   │   ├── model.py
│   │   └── README_HITL.md
│   ├── knowledge-graph/             # Knowledge graph processing
│   │   ├── create_cytoscape_files.py
│   │   ├── download_nodes.py
│   │   ├── download.py
│   │   ├── extract.py
│   │   ├── synthetic_llm.py
│   │   ├── synthetic_random.py
│   │   └── triples_to_csv.py
│   └── ux/                          # User experience components
│       ├── chat.py
│       └── select_edges_for_review.py
├── Edge_Assignor.ipynb              # Main RAG pipeline notebook
├── main.py                          # Main application entry point
├── logo.svg
├── pyproject.toml                   # Python project configuration
└── README.md                        # Project documentation
````


## Detailed Explanation of Our Methods

### Data Processing Pipeline
1. **Knowledge Graph Extraction**: Download subgraphs from the Monarch Knowledge Graph, including node metadata (identifiers, labels, descriptions) - *src/knowledge-graph/download.py*
2. **Data Preprocessing**: Convert graph triples from JSON to structured CSV format for analysis - *src/knowledge-graph/triples_to_csv.py*
4. **Edge Removal & Assignment Methodologies**: Systematically remove a percentage of edges from trusted graph data to create incomplete subgraphs. We used three strategies for creating our test KGs. - *Edge_Assignor.ipynb*. See [README-Edge_Assignor.md](README-Edge_Assignor.md) for details on RAG pipeline.
	1. **Random Baseline**:
		- Randomly assigns predicates from the set of unique relationships in the dataset
		- Provides baseline performance metrics for comparison
    2. **LLM-Based Assignment**:
        - Utilizes AWS Bedrock with OpenAI GPT models
        - Batch processing for efficiency
        - Context-aware predicate selection based on subject-object relationships
    3. **RAG-Enhanced Assignment**:
        - Integrates domain knowledge from PubMed abstracts
        - Uses ChromaDB for vector similarity search
        - Provides scientific context for relationship prediction
        - Includes PMID citations for traceability
5. **Simulated Human Curation**: A Python script that simulates human review by comparing assigned edges against ground truth, generating curated datasets for GNN training - *src/human_simulator.py*
6 **Prepare cytoscape visualization files**: Create files for visualization in Cytoscape with node & edge data for each rebuilt knowledge graph & associated backbones - *src/knowledge-graph/create_cytoscape_files.py*


### Model training & validation pipeline
1. **Validation Framework**: Compare predicted edges against ground truth using exact matching and validation scoring
2. **Graph Neural Network Training**: Extract graph backbones for GNN input and training on validation patterns


### Graph Neural Network Training

1. **Extract Graph Backbone**:
```bash
python src/knowledge-graph/extract.py
```

2. **Prepare Training Data** (see [src/gnn/README_HITL.md](src/gnn/README_HITL.md) for details):
```bash
python src/gnn/run_hitl_prep.sh
```

3. **Train GNN Model** (see [src/gnn/lcilp/README.md](src/gnn/lcilp/README.md) for details):
```bash
python src/gnn/lcilp/train.py
```


### Validation and Evaluation

- **Ground Truth Comparison**: Systematic comparison against trusted Monarch KG data
- **Accuracy Metrics**: Predicate matching rates, precision, and recall calculations
- **Error Analysis**: Categorization of prediction errors and failure modes
- **Human Validation Interface**: Prototype of an interactive web browser tool to collect expert review and feedback


### Graphical User Interface
We built a GUI!


## Future Directions
We built this prototype over 3 days as a hackathon team. We're stoked about it and are considering extending it in the future. We welcome any contributors or folks who wants to continue building off our proof-of-concept. 

## Contributing

We welcome contributions from the biomedical informatics and AI research communities. Please submit feedback and requests as 'issues'!

## License

See LICENSE file.

## Acknowledgements
The KG Model Garbage Collection tool uses and displays data and algorithms from the Monarch Initiative. The Monarch Initiative (https://monarchinitiative.org) makes biomedical knowledge exploration more efficient and effective by providing tools for genotype-phenotype analysis, genomic diagnostics, and precision medicine across broad areas of disease. We acknowledge the contributions of domain experts and the broader biomedical informatics community.
