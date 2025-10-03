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
## What can you do with KG Garbage Model Collector?
* You can collaboratively find and remove problem edges.
* Isolate part of a large knowledge graph to curate a smaller, workable data set
* Teach a GNN to find problems and curate only the problems it identifies
* Check the problems manually - iterate until agreement on what to prune

### Prerequisites
TODO

### Order of execution
TODO update with filetree
1. **src/knowledge-graph/download.py** - download a subgraph from Monarch KG (and node data including id, label, & description)
2. **src/knowledge-graph/triples_to_csv.py** - convert the downloaded triples from JSON to CSV file
3. **Edge_Assignore.ipynb** - randomly remove some edges from the downloaded triples and use 3 strategies to rebuild the edges (random, LLM, LLM-RAG)
4. **src/knowledge-graph/extract.py** - extract the "backbone" of the graph for input to GNN
5. **src/knowledge-graph/create_cytoscape_files.py** - create files for visualization in Cytoscape with node & edge data for each rebuilt knowledge graph & associated backbones


## Methods

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

#### Random Baseline
- Randomly assigns predicates from the set of unique relationships in the dataset
- Provides baseline performance metrics for comparison

#### LLM-Based Assignment
- Utilizes AWS Bedrock with OpenAI GPT models
- Batch processing for efficiency
- Context-aware predicate selection based on subject-object relationships

#### RAG-Enhanced Assignment
- Integrates domain knowledge from PubMed abstracts
- Uses ChromaDB for vector similarity search
- Provides scientific context for relationship prediction
- Includes PMID citations for traceability

### Validation and Evaluation

- **Ground Truth Comparison**: Systematic comparison against trusted Monarch KG data
- **Accuracy Metrics**: Predicate matching rates, precision, and recall calculations
- **Error Analysis**: Categorization of prediction errors and failure modes
- **Human Validation Interface**: Web-based tools for expert review and feedback collection


### RAG Pipeline for Biomedical Knowledge Graphs

This repository implements a Retrieval-Augmented Generation (RAG) workflow that:
1. Fetches abstracts from PubMed (via NCBI E-utilities)
2. Cleans and embeds them using Sentence Transformers
3. Stores embeddings in ChromaDB for efficient similarity search
4. Queries domain knowledge to assist in filling missing predicates in biomedical knowledge graphs
5. Uses AWS Bedrock LLMs with retrieved context to generate reasoning-based explanations


### AWS Bedrock Setup

Configure AWS credentials in `~/.aws/credentials`:

```ini
[default]
aws_access_key_id=YOUR_KEY
aws_secret_access_key=YOUR_SECRET
region=us-east-1
```

The script uses:
- **Model**: `openai.gpt-oss-120b-1:0`
- **Parameters**: `temperature=0.2`, `top_p=0.9`, and high `max_tokens`


### Features

- **PubMed Retrieval** – fetch abstracts using `esearch` + `efetch`
- **Abstract Cleaning** – removes affiliations and emails
- **Chunking and Embeddings** – uses `RecursiveCharacterTextSplitter` and `all-MiniLM-L6-v2`
- **Vector Database** – stores and queries embeddings in ChromaDB
- **RAG Querying** – retrieves top-k chunks for domain context
- **Predicate Filling** – enhances triples with missing predicates using LLMs
- **Evidence Generation** – produces reasoning sentences with PubMed IDs for traceability


## RAG Usage

### 1. Retrieve and Store PubMed Abstracts

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

### 2. Query Domain Knowledge

```python
results = query_domain_knowledge(
    "amyloid beta and alzheimers relationship", 
    collection, 
    model, 
    top_k=3
)
print(results)
```

### 3. Fill Missing Predicates in Triples

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


## Data Sources
The KG Model Garbage Collection tool uses and displays data and algorithms from the Monarch Initiative. The Monarch Initiative (https://monarchinitiative.org) makes biomedical knowledge exploration more efficient and effective by providing tools for genotype-phenotype analysis, genomic diagnostics, and precision medicine across broad areas of disease.

