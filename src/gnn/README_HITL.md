# HITL Preparation Workflow - Quick Start Guide

This guide covers the pre-HITL training workflow for preparing edges for human review.

---

## Quick Start (One Command)

```bash
cd /Users/yibeichen/Documents/GitHub/Model_Garbage_Collection/src/gnn
./run_hitl_prep.sh 0 50 10
```

**Arguments:**
- `0` - Iteration number (use 0 for first run)
- `50` - Number of training epochs
- `10` - Number of edges to select for review

**What it does:**
1. Extracts and splits data (if not done)
2. Trains GNN for 50 epochs
3. Scores validation pool (29 edges)
4. Selects 10 most uncertain edges for review

**Expected runtime:** ~30-40 minutes (depends on hardware)

---

## Step-by-Step (Manual)

### Step 1: Data Preparation

```bash
cd /Users/yibeichen/Documents/GitHub/Model_Garbage_Collection/src/gnn
/opt/homebrew/bin/micromamba run -n gnn-kg python extract.py
```

**Output:**
```
./lcilp/data/AlzheimersKG/focused/
├── train.txt (135 edges)
├── valid.txt (28 edges)
├── test.txt (30 edges)
└── validation_pool.txt (29 edges)
```

**Verification:**
```bash
wc -l ./lcilp/data/AlzheimersKG/focused/*
# Should show: 135 + 28 + 30 + 29 = 222 total
```

---

### Step 2: Train GNN

```bash
cd /Users/yibeichen/Documents/GitHub/Model_Garbage_Collection/src/gnn/lcilp
/opt/homebrew/bin/micromamba run -n gnn-kg python train.py \
  -d AlzheimersKG/focused \
  -e hitl_iter0 \
  --hop 3 \
  --num_gcn_layers 3 \
  --num_epochs 50 \
  --batch_size 8
```

**Key Parameters:**
- `-d AlzheimersKG/focused` - Use the focused dataset
- `-e hitl_iter0` - Experiment name (iteration 0)
- `--hop 3` - 3-hop subgraph extraction
- `--num_gcn_layers 3` - Number of GCN layers (should match hop)
- `--num_epochs 50` - Training epochs
- `--batch_size 8` - Batch size (reduce if OOM)

**Output:**
```
experiments/hitl_iter0/
├── best_graph_classifier.pth  # Trained model
├── train.log                   # Training logs
└── config.json                 # Hyperparameters
```

**Watch for:**
- Training loss should decrease over epochs
- Validation AUC should be reported
- If CUDA OOM, reduce batch size to 4

**Quick test (10 epochs):**
```bash
# Fast test to verify training works
/opt/homebrew/bin/micromamba run -n gnn-kg python train.py \
  -d AlzheimersKG/focused \
  -e test_focused \
  --hop 3 \
  --num_gcn_layers 3 \
  --num_epochs 10 \
  --batch_size 8
```

---

### Step 3: Score Validation Pool

```bash
cd /Users/yibeichen/Documents/GitHub/Model_Garbage_Collection/src/gnn/lcilp
/opt/homebrew/bin/micromamba run -n gnn-kg python score_edges.py \
  -e hitl_iter0 \
  -d AlzheimersKG/focused \
  --pool_file data/AlzheimersKG/focused/validation_pool.txt \
  --output ../../data/hitl/iteration_1/pool_scores.jsonl \
  --hop 3 \
  --batch_size 8
```

**Output:** `data/hitl/iteration_1/pool_scores.jsonl`

**Format:**
```json
{"edge_id": "MONDO:0004975_biolink:has_phenotype_HP:0002511", "triplet": ["MONDO:0004975", "biolink:has_phenotype", "HP:0002511"], "score": 0.752}
```

**Verification:**
```bash
# Check scores are valid probabilities (0-1)
cat ../../data/hitl/iteration_1/pool_scores.jsonl | jq '.score' | sort -n
```

---

### Step 4: Select Edges for Review

```bash
cd /Users/yibeichen/Documents/GitHub/Model_Garbage_Collection/src/ux
/opt/homebrew/bin/micromamba run -n gnn-kg python select_edges_for_review.py \
  --scores ../data/hitl/iteration_1/pool_scores.jsonl \
  --num_samples 10 \
  --strategy uncertainty \
  --output ../data/hitl/iteration_1/edges_to_review.jsonl
```

**Selection Strategies:**
- `uncertainty` - Select edges closest to score 0.5 (most uncertain)
- `random` - Random sample
- `low_confidence` - Select edges with lowest scores

**Output:** `data/hitl/iteration_1/edges_to_review.jsonl`

**Format:**
```json
{
  "edge_id": "MONDO:0004975_biolink:has_phenotype_HP:0002511",
  "triplet": ["MONDO:0004975", "biolink:has_phenotype", "HP:0002511"],
  "model_score": 0.52,
  "uncertainty": 0.96,
  "status": "pending"
}
```

**Verification:**
```bash
# Check selected edges
cat ../data/hitl/iteration_1/edges_to_review.jsonl | jq .
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution:**
```bash
/opt/homebrew/bin/micromamba run -n gnn-kg pip install pandas
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size
```bash
# In train.py or score_edges.py, change:
--batch_size 4  # or even 2
```

### Issue: "FileNotFoundError: data/AlzheimersKG/focused/train.txt"

**Solution:** Run data preparation first
```bash
cd /Users/yibeichen/Documents/GitHub/Model_Garbage_Collection/src/gnn
/opt/homebrew/bin/micromamba run -n gnn-kg python extract.py
```

### Issue: Training loss is NaN

**Solution:** Reduce learning rate
```bash
# Add to train.py command:
--lr 0.001
```

### Issue: "No such file or directory: experiments/hitl_iter0/best_graph_classifier.pth"

**Solution:** Training didn't complete. Check `experiments/hitl_iter0/train.log` for errors.

---

## Output Directory Structure

After running the full workflow:

```
Model_Garbage_Collection/
├── src/gnn/lcilp/
│   ├── data/AlzheimersKG/focused/
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   ├── test.txt
│   │   └── validation_pool.txt
│   ├── experiments/hitl_iter0/
│   │   ├── best_graph_classifier.pth
│   │   ├── train.log
│   │   └── config.json
│   └── data/AlzheimersKG/focused/
│       └── validation_pool_subgraphs_hitl_iter0/  # Cached subgraphs
└── data/hitl/iteration_1/
    ├── pool_scores.jsonl           # All 29 edges scored
    └── edges_to_review.jsonl       # Top 10 selected for review
```

---

## Next Steps

After running this workflow:

1. **Review the selected edges:**
   ```bash
   cat data/hitl/iteration_1/edges_to_review.jsonl | jq .
   ```

2. **Inspect score distribution:**
   ```bash
   cat data/hitl/iteration_1/pool_scores.jsonl | jq '.score' | python -c "
   import sys, statistics
   scores = [float(x) for x in sys.stdin]
   print(f'Mean: {statistics.mean(scores):.3f}')
   print(f'Stdev: {statistics.stdev(scores):.3f}')
   print(f'Min: {min(scores):.3f}')
   print(f'Max: {max(scores):.3f}')
   "
   ```

3. **Build validation UI** (coming soon)
   - Display selected edges with human-readable labels
   - Collect Approve/Reject/Skip votes
   - Export to `reviews.jsonl`

4. **Aggregate reviews** (coming soon)
   - Combine votes from multiple reviewers
   - Calculate confidence scores
   - Export to `aggregated.jsonl`

5. **Retrain with HITL** (coming soon)
   - Use aggregated scores as training weights
   - Run iteration 2

---

## Performance Expectations

### Training (50 epochs, 135 edges)
- **Time:** ~10-20 minutes
- **GPU:** Optional (CPU is sufficient for this small dataset)
- **Memory:** ~2-4 GB RAM

### Scoring (29 edges)
- **Time:** ~2-5 minutes
- **GPU:** Optional
- **Memory:** ~1-2 GB RAM

### Full Workflow
- **Time:** ~15-30 minutes total
- **Bottleneck:** Training (most time)

---

## Tips

1. **Start with quick test:** Run with 10 epochs first to verify everything works
   ```bash
   ./run_hitl_prep.sh 0 10 5
   ```

2. **Monitor training:** Check `experiments/hitl_iter0/train.log` for progress
   ```bash
   tail -f experiments/hitl_iter0/train.log
   ```

3. **Use CPU if no GPU:** Add `--gpu -1` to force CPU mode
   ```bash
   python train.py -d AlzheimersKG/focused -e test --gpu -1 ...
   ```

4. **Experiment with hyperparameters:**
   - Larger `--hop` (e.g., 4) captures more context but slower
   - More `--num_epochs` may improve performance but risk overfitting
   - Smaller `--batch_size` reduces memory but slower training

---

## Questions?

See implementation details in:
- `internal/pre-hitl-training-workflow.md` - Full workflow plan
- `internal/pre-hitl-implementation-summary.md` - What was built
- `CLAUDE.md` - Project overview and architecture
