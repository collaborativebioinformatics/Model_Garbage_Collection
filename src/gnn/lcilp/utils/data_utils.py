import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def process_files(files, saved_relation2id=None):
    """
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    """
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split("\n")[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append(
                    [
                        entity2id[triplet[0]],
                        entity2id[triplet[2]],
                        relation2id[triplet[1]],
                    ]
                )

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to each relation.
    # Merge all training data (original_train + synthetic_train_*) for adjacency list
    all_train_triplets = []
    for file_type, file_triplets in triplets.items():
        # Include 'train', 'original_train', and any 'synthetic_train_*' files
        if (
            file_type == "train"
            or file_type == "original_train"
            or file_type.startswith("synthetic_train")
        ):
            all_train_triplets.append(file_triplets)

    # Concatenate all training triplets
    if len(all_train_triplets) > 0:
        merged_train = np.concatenate(all_train_triplets, axis=0)
    else:
        # Fallback to just 'train' if no matches found
        merged_train = triplets.get("train", np.array([]))

    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(merged_train[:, 2] == i)
        adj_list.append(
            csc_matrix(
                (
                    np.ones(len(idx), dtype=np.uint8),
                    (
                        merged_train[:, 0][idx].squeeze(1),
                        merged_train[:, 1][idx].squeeze(1),
                    ),
                ),
                shape=(len(entity2id), len(entity2id)),
            )
        )

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write("\t".join([id2entity[s], id2relation[r], id2entity[o]]) + "\n")
