### RAG Pipeline for Biomedical Knowledge Graphs

This repository implements a Retrieval-Augmented Generation (RAG) workflow that:
1. Fetches abstracts from PubMed (via NCBI E-utilities)
2. leans and embeds them using Sentence Transformers
3. Stores embeddings in ChromaDB for efficient similarity search
4. Queries domain knowledge to assist in filling missing predicates in biomedical knowledge graphs
5. Uses AWS Bedrock LLMs with retrieved context to generate reasoning-based explanations

We detail the RAG usage below: 

#### Steps 1-3. Retrieve and Store PubMed Abstracts

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

#### Step 4. Query Domain Knowledge

```python
results = query_domain_knowledge(
    "amyloid beta and alzheimers relationship", 
    collection, 
    model, 
    top_k=3
)
print(results)
```

#### Step 5. Fill Missing Predicates in Triples

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