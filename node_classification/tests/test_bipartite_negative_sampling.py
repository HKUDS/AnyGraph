import sys
import types
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_DIR = REPO_ROOT / "node_classification"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(NODE_DIR))

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

_original_argv = sys.argv
sys.argv = [_original_argv[0]]
try:
    from data_handler import TrnData
finally:
    sys.argv = _original_argv


def test_node_classification_negatives_stay_in_class_partition():
    # Two graph nodes and three classes; class node IDs are offset by 2.
    train_labels = coo_matrix(
        (np.ones(2, dtype=np.float32), ([0, 1], [0, 1])), shape=(2, 3)
    )
    dataset = TrnData(train_labels)

    np.random.seed(0)
    dataset.neg_sampling()

    assert np.all(dataset.negs >= 2)
    assert np.all(dataset.negs < 5)


def test_square_graph_negatives_stay_in_node_range():
    adjacency = coo_matrix(
        (np.ones(2, dtype=np.float32), ([0, 1], [1, 2])), shape=(3, 3)
    )
    dataset = TrnData(adjacency)

    np.random.seed(0)
    dataset.neg_sampling()

    assert np.all(dataset.negs >= 0)
    assert np.all(dataset.negs < 3)
