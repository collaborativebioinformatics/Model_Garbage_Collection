import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn import metrics


class Trainer:
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        # Initialize margin ranking loss with margin γ
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info(
            "Total number of parameters: %d"
            % sum(map(lambda x: x.numel(), model_params))
        )

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(
                model_params,
                lr=params.lr,
                momentum=params.momentum,
                weight_decay=self.params.l2,
            )
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(
                model_params, lr=params.lr, weight_decay=self.params.l2
            )

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction="sum")

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(
            self.train_data,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers,
            collate_fn=self.params.collate_fn,
        )
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        for b_idx, batch in enumerate(dataloader):
            # Move batch to device (GPU/CPU)
            data_pos, targets_pos, data_neg, targets_neg = (
                self.params.move_batch_to_device(batch, self.params.device)
            )
            self.optimizer.zero_grad()

            # Step 1: Score positive triplets: # Score(t+)
            score_pos = self.graph_classifier(data_pos)

            # Step 2: Score negative triplets (multiple negatives per positive): # Score(t-)
            score_neg = self.graph_classifier(data_neg)
            # Shape: (batch_size * num_neg_samples_per_link, 1)

            # Step 3: Compute margin ranking loss
            # For each positive, average the scores of its negative samples
            # ... score_neg_mean = score_neg.view(len(score_pos), -1).mean(dim=1)

            # MarginRankingLoss computes: max(0, margin - score_pos + score_neg_mean)
            # The "1" tensor indicates we want score_pos > score_neg_mean

            # TODO: insert HITL biases here:
            # if yes/check: reduce loss
            # if no/X: increase loss
            # if nothing: don't change loss
            # Squeeze score_pos to match score_neg dimensions: [batch_size, 1] -> [batch_size]
            # Create target tensor with same batch size: [1] -> [batch_size]
            batch_size = len(score_pos)
            loss = self.criterion(
                score_pos.squeeze(1),
                score_neg.view(batch_size, -1).mean(dim=1),
                torch.ones(batch_size).to(device=self.params.device),
            )

            # BackProp
            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            # performance evaluation metrics
            with torch.no_grad():
                all_scores += (
                    score_pos.squeeze().detach().cpu().tolist()
                    + score_neg.squeeze().detach().cpu().tolist()
                )
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss

            if (
                self.valid_evaluator
                and self.params.eval_every_iter
                and self.updates_counter % self.params.eval_every_iter == 0
            ):
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info(
                    "\nPerformance:" + str(result) + "in " + str(time.time() - tic)
                )

                if result["auc"] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result["auc"]
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(
                            f"Validation performance didn't improve for {self.params.early_stop} epochs. Training stops."
                        )
                        break
                self.last_metric = result["auc"]

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info(
                f"Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}"
            )

            # if self.valid_evaluator and epoch % self.params.eval_every == 0:
            #     result = self.valid_evaluator.eval()
            #     logging.info('\nPerformance:' + str(result))

            #     if result['auc'] >= self.best_metric:
            #         self.save_classifier()
            #         self.best_metric = result['auc']
            #         self.not_improved_count = 0

            #     else:
            #         self.not_improved_count += 1
            #         if self.not_improved_count > self.params.early_stop:
            #             logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
            #             break
            #     self.last_metric = result['auc']

            if epoch % self.params.save_every == 0:
                torch.save(
                    self.graph_classifier,
                    os.path.join(self.params.exp_dir, "graph_classifier_chk.pth"),
                )

    def save_classifier(self):
        torch.save(
            self.graph_classifier,
            os.path.join(self.params.exp_dir, "best_graph_classifier.pth"),
        )  # Does it overwrite or fuck with the existing file?
        logging.info("Better models found w.r.t accuracy. Saved it!")
