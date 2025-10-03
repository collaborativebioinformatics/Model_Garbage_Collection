# Production Pipeline Usage Guide

**Last Updated:** 2025-10-03  
**Branch:** `production-data-pipeline`

---

## Overview

This guide documents how to use the production data pipeline that combines **original** (real) edges with **synthetic** (LLM/RAG-generated) edges for HITL training.

## Data Structure

```
src/gnn/lcilp/data/AlzheimersKG/
├── original/
│   ├── train.txt          # Real Monarch edges for training (135 edges)
│   ├── valid.txt          # Real Monarch edges for validation (28 edges)
│   └── test.txt           # Real Monarch edges for testing (30 edges)
│
├── synthetic/
│   ├── train_random.txt   # Synthetic edges for training (135 edges)
│   ├── valid_random.txt   # Synthetic edges for validation/review (28 edges)
│   └── test_random.txt    # Synthetic edges for testing (30 edges)
│
└── focused/ (legacy MVP structure, still supported)
    ├── train.txt
    ├── valid.txt
    ├── test.txt
    └── validation_pool.txt
```

---

## Commands

### 1. Training with Production Data

Train GNN on **original + synthetic** edges:

```bash
cd src/gnn/lcilp

# Production mode: combines original/train.txt + synthetic/train_random.txt
/opt/homebrew/bin/micromamba run -n gnn-kg python train.py \
  -d AlzheimersKG \
  -e hitl_iter0_production \
  --use_production_data \
  --synthetic_train_files train_random.txt \
  --hop 3 \
  --num_gcn_layers 3 \
  --num_epochs 50 \
  --batch_size 8
```

**Key flags:**
- `--use_production_data`: Enable production mode (original/ + synthetic/)
- `--synthetic_train_files`: Comma-separated synthetic files (e.g., `train_random.txt,train_llm.txt,train_rag.txt`)
- `-d AlzheimersKG`: Dataset name (no "focused" suffix in production mode)

### 2. Backward Compatible MVP Mode

Train on focused dataset only (no synthetic edges):

```bash
cd src/gnn/lcilp

# MVP mode: uses focused/train.txt only
/opt/homebrew/bin/micromamba run -n gnn-kg python train.py \
  -d AlzheimersKG/focused \
  -e hitl_iter0_mvp \
  --hop 3 \
  --num_gcn_layers 3 \
  --num_epochs 50 \
  --batch_size 8
```

**Note:** Omitting `--use_production_data` defaults to MVP mode.

---

### 3. Scoring Synthetic Validation Edges

Score **synthetic/valid_*.txt** for human review:

```bash
cd src/gnn/lcilp

# Production mode: score synthetic validation edges
/opt/homebrew/bin/micromamba run -n gnn-kg python score_edges.py \
  --experiment_name hitl_iter0_production \
  --dataset AlzheimersKG \
  --synthetic_files synthetic/valid_random.txt \
  --output ../../data/hitl/iteration_1/synthetic_valid_scores.jsonl
```

**Key flags:**
- `--synthetic_files`: Comma-separated synthetic validation files
- `--dataset AlzheimersKG`: No "focused" suffix

### 4. Backward Compatible Scoring (MVP)

Score validation pool (29 backbone edges):

```bash
cd src/gnn/lcilp

# MVP mode: score validation_pool.txt
/opt/homebrew/bin/micromamba run -n gnn-kg python score_edges.py \
  --experiment_name hitl_iter0_mvp \
  --dataset AlzheimersKG/focused \
  --pool_file focused/validation_pool.txt \
  --output ../../data/hitl/iteration_1/pool_scores.jsonl
```

---

## Data Flow

### Production HITL Workflow

```
1. Generate synthetic edges
   └─> synthetic/train_random.txt, valid_random.txt, test_random.txt

2. Train GNN (iteration 0)
   ├─ Input: original/train.txt + synthetic/train_random.txt
   ├─ Validation: original/valid.txt
   └─> Model: experiments/hitl_iter0_production/

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
   └─> Model: experiments/hitl_iter1_production/

8. Evaluate on test set
   ├─ Input: synthetic/test_random.txt
   └─> Compare against: original/test.txt (ground truth)
```

---

## Key Differences: Production vs MVP

| Aspect | Production Mode | MVP Mode (Legacy) |
|--------|----------------|-------------------|
| **Flag** | `--use_production_data` | Default (no flag) |
| **Dataset path** | `AlzheimersKG` | `AlzheimersKG/focused` |
| **Training data** | `original/train.txt` + `synthetic/train_*.txt` | `focused/train.txt` |
| **Validation** | `original/valid.txt` | `focused/valid.txt` |
| **Review pool** | `synthetic/valid_*.txt` | `focused/validation_pool.txt` |
| **Scoring** | `--synthetic_files` | `--pool_file` |
| **Purpose** | Full HITL pipeline | MVP testing |

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

## Troubleshooting

### "FileNotFoundError: original/train.txt"

**Cause:** Using `--use_production_data` with old dataset structure.

**Fix:** Ensure `original/` and `synthetic/` directories exist:
```bash
ls src/gnn/lcilp/data/AlzheimersKG/
# Should show: original/ synthetic/ focused/
```

### "Either --synthetic_files or --pool_file must be provided"

**Cause:** `score_edges.py` requires one of these flags.

**Fix:**
- Production: Add `--synthetic_files synthetic/valid_random.txt`
- MVP: Add `--pool_file focused/validation_pool.txt`

### Training uses only original edges, ignoring synthetic

**Cause:** Forgot `--use_production_data` flag.

**Fix:** Add the flag to enable production mode.

---

## Next Steps

1. **Run production training** to verify combined original + synthetic edges work
2. **Score synthetic validation edges** to test active learning selection
3. **Implement validation UI** for human review (coming soon)
4. **Replace mock synthetic edges** with real LLM/RAG output when ready

---

## References

- **HITL System Design:** `/internal/hitl-system-design.md`
- **MVP Implementation Plan:** `/internal/hitl-mvp-focused-dataset-plan.md`
- **Data Preparation:** `/src/gnn/extract.py`, `/src/gnn/generate_synthetic_edges.py`
