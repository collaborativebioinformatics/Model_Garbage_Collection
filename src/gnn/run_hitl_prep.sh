#!/bin/bash
set -e  # Exit on error

ITERATION=${1:-0}
NUM_EPOCHS=${2:-50}
NUM_EDGES_TO_REVIEW=${3:-10}

echo "============================================"
echo "HITL Preparation Workflow"
echo "Iteration: $ITERATION"
echo "Epochs: $NUM_EPOCHS"
echo "Edges to review: $NUM_EDGES_TO_REVIEW"
echo "============================================"

# Step 1: Extract data (only if needed)
if [ ! -f "./lcilp/data/AlzheimersKG/focused/train.txt" ]; then
    echo -e "\n[1/5] Extracting and splitting data..."
    cd /Users/yibeichen/Documents/GitHub/Model_Garbage_Collection/src/gnn
    /opt/homebrew/bin/micromamba run -n gnn-kg python extract.py
else
    echo -e "\n[1/5] Data already extracted, skipping..."
fi

# Step 2: Train GNN
echo -e "\n[2/5] Training GNN..."
cd /Users/yibeichen/Documents/GitHub/Model_Garbage_Collection/src/gnn/lcilp
/opt/homebrew/bin/micromamba run -n gnn-kg python train.py \
  -d AlzheimersKG/focused \
  -e hitl_iter${ITERATION} \
  --hop 3 \
  --num_gcn_layers 3 \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size 8

# Step 3: Score validation pool
echo -e "\n[3/5] Scoring validation pool..."
/opt/homebrew/bin/micromamba run -n gnn-kg python score_edges.py \
  -e hitl_iter${ITERATION} \
  -d AlzheimersKG/focused \
  --pool_file data/AlzheimersKG/focused/validation_pool.txt \
  --output ../../data/hitl/iteration_$((ITERATION+1))/pool_scores.jsonl \
  --hop 3 \
  --batch_size 8

# Step 4: Select edges for review
echo -e "\n[4/5] Selecting edges for human review..."
cd /Users/yibeichen/Documents/GitHub/Model_Garbage_Collection/src/ux
/opt/homebrew/bin/micromamba run -n gnn-kg python select_edges_for_review.py \
  --scores ../data/hitl/iteration_$((ITERATION+1))/pool_scores.jsonl \
  --num_samples ${NUM_EDGES_TO_REVIEW} \
  --strategy uncertainty \
  --output ../data/hitl/iteration_$((ITERATION+1))/edges_to_review.jsonl

# Step 5: Summary
echo -e "\n[5/5] Workflow complete!"
echo "============================================"
echo "Results:"
echo "- Model: src/gnn/lcilp/experiments/hitl_iter${ITERATION}/"
echo "- Scores: data/hitl/iteration_$((ITERATION+1))/pool_scores.jsonl"
echo "- Edges to review: data/hitl/iteration_$((ITERATION+1))/edges_to_review.jsonl"
echo "============================================"
echo "Next steps:"
echo "1. Review scores: cat data/hitl/iteration_$((ITERATION+1))/pool_scores.jsonl | head -5"
echo "2. Review selected edges: cat data/hitl/iteration_$((ITERATION+1))/edges_to_review.jsonl"
echo "3. Launch validation UI (coming soon)"
echo "============================================"
