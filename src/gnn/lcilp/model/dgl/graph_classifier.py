from .rgcn_model import RGCN

# MIGRATION: Updated for PyTorch 2.1.2/DGL 1.1.3
import dgl
import torch.nn as nn
import torch

"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(
        self, params, relation2id
    ):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(
            self.params.num_rels, self.params.rel_emb_dim, sparse=False
        )

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(
                3 * self.params.num_gcn_layers * self.params.emb_dim
                + self.params.rel_emb_dim,
                1,
            )
        else:
            self.fc_layer = nn.Linear(
                self.params.num_gcn_layers * self.params.emb_dim
                + self.params.rel_emb_dim,
                1,
            )

    def forward(self, data):
        g, rel_labels = data

        # get node embeddings
        g.ndata["h"] = self.gnn(g)

        # average pooling: g_S = (1/|V_S|) Σ h_v
        # MIGRATION: Updated for PyTorch 2.1.2/DGL 1.1.3
        g_out = dgl.readout_nodes(g, "repr", op="mean")  # Shape: (batch, L * emb_dim)

        # Extract head and tail node embeddings
        head_ids = (g.ndata["id"] == 1).nonzero().squeeze(1)
        head_embs = g.ndata["repr"][head_ids]  # Shape: (batch, L * emb_dim)

        tail_ids = (g.ndata["id"] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata["repr"][tail_ids]  # Shape: (batch, L * emb_dim)

        if self.params.add_ht_emb:
            g_rep = torch.cat(
                [
                    g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                    head_embs.view(
                        -1, self.params.num_gcn_layers * self.params.emb_dim
                    ),
                    tail_embs.view(
                        -1, self.params.num_gcn_layers * self.params.emb_dim
                    ),
                    self.rel_emb(rel_labels),
                ],
                dim=1,
            )
        else:
            g_rep = torch.cat(
                [
                    g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                    self.rel_emb(rel_labels),
                ],
                dim=1,
            )

        output = self.fc_layer(g_rep)
        return output
