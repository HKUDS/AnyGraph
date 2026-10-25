import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.sparse import coo_matrix


# data_handler imports graph transforms, which these unit tests do not exercise.
try:
    import torch_geometric.transforms  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "torch_geometric":
        raise
    torch_geometric = types.ModuleType("torch_geometric")
    torch_geometric.__path__ = []
    torch_geometric.transforms = types.ModuleType("torch_geometric.transforms")
    sys.modules["torch_geometric"] = torch_geometric
    sys.modules["torch_geometric.transforms"] = torch_geometric.transforms

# params.py parses CLI arguments at import time; keep pytest flags out of it.
_original_argv = sys.argv
sys.argv = [_original_argv[0]]
try:
    from data_handler import TrnData
finally:
    sys.argv = _original_argv


def test_bipartite_negative_samples_are_candidate_items():
    # Two users, three items; positive item IDs are offset by the user count.
    interactions = coo_matrix(
        (np.ones(2, dtype=np.float32), ([0, 1], [0, 1])), shape=(2, 3)
    )
    dataset = TrnData(interactions)

    np.random.seed(0)
    dataset.neg_sampling()

    assert np.all(dataset.negs >= 2)
    assert np.all(dataset.negs < 5)


def test_square_graph_negative_samples_remain_in_node_range():
    adjacency = coo_matrix(
        (np.ones(2, dtype=np.float32), ([0, 1], [1, 2])), shape=(3, 3)
    )
    dataset = TrnData(adjacency)

    np.random.seed(0)
    dataset.neg_sampling()

    assert np.all(dataset.negs >= 0)
    assert np.all(dataset.negs < 3)
