#!/usr/bin/env bash


DATASET=$1
EXP_NAME=$2
HOP=$3
EPOCH=$4


python train.py -d "$DATASET" -e "$DATASET"_"$EXP_NAME"_"$HOP"_"$EPOCH" --hop $HOP --num_gcn_layers $HOP --num_epochs $EPOCH

python test_ranking.py -d "$DATASET"_ind -e "$DATASET"_"$EXP_NAME"_"$HOP"_"$EPOCH" --hop $HOP

python test_auc.py -d "$DATASET"_ind -e "$DATASET"_"$EXP_NAME"_"$HOP"_"$EPOCH" --hop $HOP