# HITL Pipeline Usage Guide

**Last Updated:** 2025-10-03  
**Branch:** `production-data-pipeline`

---

## Overview

This guide documents how to use the HITL (Human-in-the-Loop) pipeline that combines **original** (real Monarch KG) edges with **synthetic** (LLM/RAG-generated) edges for iterative GNN training.

## Data Structure

```
src/gnn/lcilp/data/AlzheimersKG/
├── original/
│   ├── train.txt          # Real Monarch edges for training (135 edges)
│   ├── valid.txt          # Real Monarch edges for validation (28 edges)
│   └── test.txt           # Real Monarch edges for testing (30 edges)
│
└── synthetic/
    ├── train_random.txt   # Synthetic edges for training (135 edges)
    ├── valid_random.txt   # Synthetic edges for validation/review (28 edges)
    └── test_random.txt    # Synthetic edges for testing (30 edges)
```

---

## Commands

### 1. Training with Original + Synthetic Data

Train GNN on **original + synthetic** edges:

```bash
cd src/gnn/lcilp

# Train on original/train.txt + synthetic/train_random.txt
/opt/homebrew/bin/micromamba run -n gnn-kg python train.py \
  -d AlzheimersKG \
  -e hitl_iter0 \
  --synthetic_train_files train_random.txt \
  --hop 3 \
  --num_gcn_layers 3 \
  --num_epochs 50 \
  --batch_size 8
```

**Key flags:**
- `-d AlzheimersKG`: Dataset name
- `--synthetic_train_files`: Comma-separated synthetic files (e.g., `train_random.txt,train_llm.txt,train_rag.txt`)
- Default: automatically loads `original/train.txt` + specified synthetic files

### 2. Scoring Synthetic Validation Edges

Score **synthetic/valid_*.txt** for human review:

```bash
cd src/gnn/lcilp

# Score synthetic validation edges
/opt/homebrew/bin/micromamba run -n gnn-kg python score_edges.py \
  --experiment_name hitl_iter0 \
  --dataset AlzheimersKG \
  --synthetic_files synthetic/valid_random.txt \
  --output ../../data/hitl/iteration_1/synthetic_valid_scores.jsonl
```

**Key flags:**
- `--synthetic_files`: Comma-separated synthetic validation files
- `--dataset AlzheimersKG`: Dataset name (no subdirectory)

---

## Data Flow

### Complete HITL Workflow

```
1. Generate synthetic edges
   └─> synthetic/train_random.txt, valid_random.txt, test_random.txt

2. Train GNN (iteration 0)
   ├─ Input: original/train.txt + synthetic/train_random.txt
   ├─ Validation: original/valid.txt
   └─> Model: experiments/hitl_iter0/

3. Score synthetic validation edges
   ├─ Input: synthetic/valid_random.txt
   └─> Output: data/hitl/iteration_1/synthetic_valid_scores.jsonl

4. Select uncertain edges for review
   ├─ Input: synthetic_valid_scores.jsonl
   └─> Output: data/hitl/iteration_1/edges_to_review.jsonl (10-15 edges)

5. Human review (via validation UI)
   ├─ Input: edges_to_review.jsonl
   ├─ Output: reviews.jsonl
   └─> Aggregated: aggregated.jsonl

6. Build augmented training set
   ├─ Combine: original/train + synthetic/train + validated edges
   └─> Output: data/hitl/iteration_1/train_augmented.jsonl

7. Retrain with HITL (iteration 1)
   ├─ Input: train_augmented.jsonl (with per-edge weights)
   └─> Model: experiments/hitl_iter1/

8. Evaluate on test set
   ├─ Input: synthetic/test_random.txt
   └─> Compare against: original/test.txt (ground truth)
```

---

## File Formats

### Edge Files (TSV)

All edge files use tab-separated triplet format:

```
MONDO:0004975	biolink:has_phenotype	HP:0000716
MONDO:0004975	biolink:subclass_of	MONDO:0005395
...
```

### Scored Edges (JSONL)

Output from `score_edges.py`:

```jsonl
{"edge_id": "MONDO:0004975_biolink:has_phenotype_HP:0000716", "triplet": ["MONDO:0004975", "biolink:has_phenotype", "HP:0000716"], "score": 0.732}
{"edge_id": "MONDO:0004975_biolink:subclass_of_MONDO:0005395", "triplet": ["MONDO:0004975", "biolink:subclass_of", "MONDO:0005395"], "score": 0.891}
```

---

## Generating Mock Synthetic Edges

For testing before LLM/RAG integration:

```bash
cd src/gnn
/opt/homebrew/bin/micromamba run -n gnn-kg python generate_synthetic_edges.py
```

This creates:
- `synthetic/train_random.txt` (30% real, 50% corrupted, 20% random)
- `synthetic/valid_random.txt`
- `synthetic/test_random.txt`

**Note:** Replace with real LLM/RAG output when available (no code changes needed, just data swap).

---

## Preparing Original Data

Run `extract.py` to create the original dataset splits:

```bash
cd src/gnn
/opt/homebrew/bin/micromamba run -n gnn-kg python extract.py
```

This extracts subgraphs from the full Alzheimer's KG and creates:
- `original/train.txt` (135 edges, 70%)
- `original/valid.txt` (28 edges, 15%)
- `original/test.txt` (30 edges, 15%)

---

## Troubleshooting

### "FileNotFoundError: original/train.txt"

**Cause:** Dataset structure not created yet.

**Fix:** Run data preparation:
```bash
cd src/gnn
/opt/homebrew/bin/micromamba run -n gnn-kg python extract.py
/opt/homebrew/bin/micromamba run -n gnn-kg python generate_synthetic_edges.py
```

### Training uses only original edges, ignoring synthetic

**Cause:** Forgot `--synthetic_train_files` flag or wrong filename.

**Fix:** Add the flag: `--synthetic_train_files train_random.txt`

### "ModuleNotFoundError: No module named 'numpy'"

**Cause:** Not using the `gnn-kg` conda environment.

**Fix:** Prefix commands with `/opt/homebrew/bin/micromamba run -n gnn-kg`

---

## Next Steps

1. **Run training** to verify combined original + synthetic edges work
2. **Score synthetic validation edges** to test active learning selection
3. **Implement validation UI** for human review (coming soon)
4. **Replace mock synthetic edges** with real LLM/RAG output when ready

---

## References

- **HITL System Design:** `/internal/hitl-system-design.md`
- **Data Preparation:** `/src/gnn/extract.py`, `/src/gnn/generate_synthetic_edges.py`
- **GNN Architecture:** `/internal/gnn-architecture.md`
