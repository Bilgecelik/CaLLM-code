import torch
import torch.nn.functional as F
from collections import defaultdict


class PrototypeManager:
    """ Class to handle a dictionary of prototypes with usage tracking. """
    def __init__(self):

        self.prototypes = defaultdict(lambda: {"embedding": torch.Tensor(),
                                               "usage_count": 0,
                                               "total_weight": []})
        self.count = 0  # Tracks the number of prototypes

    def add_prototype(self, new_embedding, initial_weight=1.0):
        """ Adds a new prototype to the dictionary. """
        self.prototypes[str(self.count)] = {
            "embedding": new_embedding,
            "usage_count": 1,
            "total_weight": [initial_weight]
        }
        self.count += 1

    def reinitialize_prototype(self, prototype_id, new_embedding, initial_weight=1.0):
        """ Re-initialize the prototype with the given id and new embedding. """
        self.prototypes[prototype_id] = {
            "embedding": new_embedding,
            "usage_count": 1,
            "total_weight": [initial_weight]
        }

    def update_prototype(self, prototype_idx, embedding, weight=1.0):
        """ Updates an existing prototype by averaging embeddings and tracking usage. """
        current_prototype = self.prototypes[prototype_idx]
        embeddings = torch.stack((embedding, current_prototype["embedding"]), dim=0)
        self.prototypes[prototype_idx]["embedding"] = embeddings.mean(dim=0)
        self.prototypes[prototype_idx]["usage_count"] += 1
        self.prototypes[prototype_idx]["total_weight"].append(weight)

    def find_closest_prototype(self, embedding, topk):
        if self.count == 0:  # If the list is empty
            return [None], [float('inf')]
        else:
            distances = {
                key: 1 - F.cosine_similarity(embedding, prototype["embedding"], dim=1).item()
                for key, prototype in self.prototypes.items()
            }

            sorted_distances = sorted(distances.items(), key=lambda item: item[1])
            sorted_indices = [item[0] for item in sorted_distances[:topk]]
            sorted_distances_values = [item[1] for item in sorted_distances[:topk]]
            return sorted_indices, sorted_distances_values
