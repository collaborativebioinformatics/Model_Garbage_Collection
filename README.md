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
Biomedical knowledge graphs (KG) are powerful tools for linking genes, diseases, and phenotypes — but when AI models generate new edges, they often hallucinate or introduce errors. Our project focuses on pruning these errors. The KG Model Garbage Collection Tool is a proof-of-concept for how a combination of human review, grounded AI, and graph learning can work together to keep biomedical knowledge graphs accurate and trustworthy.

The KG Model Garbage Collection Tool uses a subset of a trusted graph (Monarch) and randomly removes some edges. Then, three strategies are used to fill the missing edges: random guessing, a general LLM, and an LLM using biomedical RAG. Participants (SMEs) validate (some of) these edges through a simple interface to evaluate how close each method comes to the truth. This data is used to train a graph neural network to see if it can automatically spot questionable edges and flag them for review and removal. The resulting knowledge graph is tested against the original, trusted knowledge graph. 

## Problem
Generating knowledge graphs at scale requires the use of Large Language Models (LLMs), which can introduce errors and hallucinations. In biomedicine, mistakes could lead to errors that cause harm to people. While Human-in-the-Loop (HITL) approaches can mitigate risks, they are not scalable solutions for large, complex knowledge graphs.

## Solution
The KG Model Garbage Collection Tool provides a proof-of-concept (PoC) framework allowing curators to probe a KG using real scientific questions, provide feedback, and use that feedback to train a GNN. This tool extends the impact of human curation by learning from expert human validation patterns.

<img width="561" height="518" alt="Screenshot 2025-10-03 at 9 23 05 AM" src="https://github.com/user-attachments/assets/933c615e-d665-4841-a5d6-1bd89adf72d0" />


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

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- pandas
- numpy
- boto3 (AWS SDK)
- sentence-transformers
- chromadb
- langchain
- requests

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
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
export AWS_REGION=us-east-1
export AWS_PROFILE=your-profile
```

### Order of Execution
Belo is a quick logic flow behind each of our scripts. 

1. **src/knowledge-graph/download.py** - download a subgraph from Monarch KG (and node data including id, label, & description)
2. **src/knowledge-graph/triples_to_csv.py** - convert the downloaded triples from JSON to CSV file
3. **Edge_Assignor.ipynb** - randomly remove some edges from the downloaded triples and use 3 strategies to rebuild the edges (random, LLM, LLM-RAG)
4. **src/knowledge-graph/extract.py** - extract the "backbone" of the graph for input to GNN
5. **src/knowledge-graph/create_cytoscape_files.py** - create files for visualization in Cytoscape with node & edge data for each rebuilt knowledge graph & associated backbones

### Directory Structure
````Model_Garbage_Collection/
├── app/
│   └── frontend/                    # React frontend application
│       ├── src/
│       │   ├── components/
│       │   │   ├── graphviewer/
│       │   │   │   └── GraphViewer.tsx
│       │   │   ├── CounterCard.tsx
│       │   │   ├── GraphView.tsx
│       │   │   ├── StatsCard.tsx
│       │   │   ├── TableView.tsx
│       │   │   ├── TriToggle.tsx
│       │   │   └── UserCard.tsx
│       │   ├── data/alzheimers_llm/
│       │   │   ├── backbone_graph.json
│       │   │   ├── graph.json
│       │   │   └── add_edge_ids.sh
│       │   ├── types/
│       │   │   └── GraphInterface.tsx
│       │   ├── App.tsx
│       │   ├── main.tsx
│       │   ├── store.ts
│       │   └── theme.ts
│       ├── index.html
│       ├── package.json
│       ├── README.md
│       ├── tsconfig.json
│       └── vite.config.ts
├── data/                            # Input datasets
│   ├── alzheimers_nodes.json
│   ├── alzheimers_triples.csv
│   └── example_edges.jsonl
├── notebooks/                       # Jupyter notebooks for analysis
│   └── model_testing.ipynb
├── outputs/                         # Generated results and datasets
│   ├── cytoscape/                   # Graph visualization files
│   │   ├── alzheimers_llm_rag-backbone.json
│   │   ├── alzheimers_llm_rag.json
│   │   ├── alzheimers_llm-backbone.json
│   │   ├── alzheimers_llm.json
│   │   ├── alzheimers_random-backbone.json
│   │   └── alzheimers_random.json
│   ├── antijoin_alzheimers_llm_rag.csv
│   ├── antijoin_alzheimers_llm.csv
│   ├── antijoin_alzheimers_random.csv
│   ├── bedrock_filled_test.csv
│   ├── bedrock_rag_filled_test.csv
│   ├── bedrock_rag_metrics_test.json
│   ├── bedrock_rag_responses_test.json
│   ├── modified_chunk_50%_removed.csv
│   └── randomly_assigned_edges.csv
├── src/                             # Core source code
│   ├── gnn/                         # Graph Neural Network components
│   │   ├── lcilp/                   # Link prediction implementation
│   │   │   ├── data/                # Training datasets
│   │   │   │   ├── FB15K237/
│   │   │   │   ├── NELL-995/
│   │   │   │   ├── WN18RR/
│   │   │   │   ├── alzheimers_*.csv
│   │   │   │   └── [various versioned datasets]
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
│   │   │   │   ├── data_utils.py
│   │   │   │   ├── dgl_utils.py
│   │   │   │   ├── graph_utils.py
│   │   │   │   └── initialization_utils.py
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
├── README.md                        # Project documentation
└── README_RAG.md                    # RAG pipeline documentation
````


## Detailed Explanation of Our Methods

### Data Processing Pipeline
1. **Knowledge Graph Extraction**: Download subgraphs from the Monarch Knowledge Graph, including node metadata (identifiers, labels, descriptions)
2. **Data Preprocessing**: Convert graph triples from JSON to structured CSV format for analysis
3. **Edge Removal**: Systematically remove a percentage of edges from trusted graph data to create incomplete subgraphs
4. **Edge Prediction Strategies**:
   - **Random Assignment**: Baseline method using random selection from available predicates
   - **LLM-Based Prediction**: AWS Bedrock API with GPT models for context-aware edge prediction
   - **RAG-Enhanced Prediction**: Retrieval-Augmented Generation using PubMed abstracts and ChromaDB for domain-specific context
5. **Validation Framework**: Compare predicted edges against ground truth using exact matching and validation scoring
6. **Graph Neural Network Training**: Extract graph backbones for GNN input and training on validation patterns

### Edge Assignment Methodologies

A. **Random Baseline**:
- Randomly assigns predicates from the set of unique relationships in the dataset
- Provides baseline performance metrics for comparison

B. **LLM-Based Assignment**:
- Utilizes AWS Bedrock with OpenAI GPT models
- Batch processing for efficiency
- Context-aware predicate selection based on subject-object relationships

C. **RAG-Enhanced Assignment**:
- Integrates domain knowledge from PubMed abstracts
- Uses ChromaDB for vector similarity search
- Provides scientific context for relationship prediction
- Includes PMID citations for traceability

### RAG Pipeline for Biomedical Knowledge Graphs

This repository implements a Retrieval-Augmented Generation (RAG) workflow that:
1. Fetches abstracts from PubMed (via NCBI E-utilities)
2. Cleans and embeds them using Sentence Transformers
3. Stores embeddings in ChromaDB for efficient similarity search
4. Queries domain knowledge to assist in filling missing predicates in biomedical knowledge graphs
5. Uses AWS Bedrock LLMs with retrieved context to generate reasoning-based explanations


### RAG Usage

#### 1. Retrieve and Store PubMed Abstracts

The pipeline automatically:
- Retrieves PubMed abstracts (default: Alzheimer's disease)
- Cleans, chunks, embeds, and stores them in ChromaDB

```python
import requests
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Initialize components
client = chromadb.Client()
collection = client.create_collection("pubtator_data")
model = SentenceTransformer("all-MiniLM-L6-v2")
```

#### 2. Query Domain Knowledge

```python
results = query_domain_knowledge(
    "amyloid beta and alzheimers relationship", 
    collection, 
    model, 
    top_k=3
)
print(results)
```

#### 3. Fill Missing Predicates in Triples

```python
filled_df, metrics, responses = fill_missing_predicates_llm_with_domain_knowledge(
    input_df=my_triples_df,
    unique_predicates=["biolink:related_to", "biolink:interacts_with", "biolink:causes"],
    collection=collection,
    model=model,
    output_file='bedrock_rag_filled_test.csv',
    metrics_file='bedrock_rag_metrics_test.json',
    responses_file='bedrock_rag_responses_test.json'
)
```

This produces:
- `bedrock_rag_filled_test.csv` – triples with filled predicates
- `bedrock_rag_metrics_test.json` – run statistics
- `bedrock_rag_responses_test.json` – detailed LLM responses


### Validation and Evaluation

- **Ground Truth Comparison**: Systematic comparison against trusted Monarch KG data
- **Accuracy Metrics**: Predicate matching rates, precision, and recall calculations
- **Error Analysis**: Categorization of prediction errors and failure modes
- **Human Validation Interface**: Web-based tools for expert review and feedback collection


### Graph Neural Network Training

1. **Extract Graph Backbone**:
```bash
python src/knowledge-graph/extract.py
```

2. **Prepare Training Data**:
```bash
python src/gnn/run_hitl_prep.sh
```

3. **Train GNN Model**:
```bash
python src/gnn/lcilp/train.py
```

### Graphical User Interface
We built a GUI!


## Future Directions

### Short-term Enhancements

1. **Expanded Model Support**: Integration with additional LLM providers (Anthropic, Cohere, local models)
2. **Advanced RAG Techniques**: Implementation of more sophisticated retrieval strategies and knowledge fusion methods
3. **Interactive Validation Interface**: Development of web-based tools for streamlined expert review
4. **Performance Optimization**: Batch processing improvements and caching mechanisms for large-scale operations

### Medium-term Research Goals

1. **Active Learning Integration**: Implement uncertainty-based sampling for targeted human validation
2. **Multi-modal Knowledge Integration**: Incorporate image, genomic, and clinical data sources
3. **Federated Learning Approaches**: Enable collaborative model training across institutions while preserving data privacy
4. **Explainable AI Methods**: Develop interpretable models for edge prediction and error detection

### Long-term Vision

1. **Real-time Quality Monitoring**: Continuous assessment of knowledge graph integrity and automated error detection
2. **Domain-specific Specialization**: Tailored models for specific biomedical subdomains (oncology, neurology, genetics)
3. **Integration with Clinical Workflows**: Direct integration with electronic health records and clinical decision support systems
4. **Standardization and Interoperability**: Development of standard protocols for knowledge graph quality assessment across platforms

### Technical Roadmap

1. **Scalability Improvements**: Architecture enhancements for processing larger knowledge graphs (millions of edges)
2. **Advanced Graph Analytics**: Implementation of graph-theoretic measures for quality assessment
3. **Automated Pipeline Orchestration**: Development of MLOps workflows for continuous model improvement
4. **Cross-validation Studies**: Large-scale validation across multiple biomedical knowledge bases

## Contributing

We welcome contributions from the biomedical informatics and AI research communities. Please submit feedback and requests as 'issues'!

## License


## Acknowledgments

This project builds upon the foundational work of the Monarch Initiative and leverages data from multiple biomedical knowledge sources. We acknowledge the contributions of domain experts and the broader biomedical informatics community.


## Data Sources
The KG Model Garbage Collection tool uses and displays data and algorithms from the Monarch Initiative. The Monarch Initiative (https://monarchinitiative.org) makes biomedical knowledge exploration more efficient and effective by providing tools for genotype-phenotype analysis, genomic diagnostics, and precision medicine across broad areas of disease.
