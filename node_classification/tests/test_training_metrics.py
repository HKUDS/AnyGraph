import sys
import types
from pathlib import Path

import pytest

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

sys.modules.setdefault("setproctitle", types.ModuleType("setproctitle"))
_original_argv = sys.argv
sys.argv = [_original_argv[0]]
try:
    import main as node_main
finally:
    sys.argv = _original_argv


class _Handler:
    data_name = "synthetic"

    def make_projectors(self):
        pass


class _Model:
    def assign_experts(self, *args, **kwargs):
        pass


class _MultiHandler:
    trn_handlers = []
    tst_handlers_group = [[_Handler()]]

    def remake_initial_projections(self):
        pass


def test_node_classification_training_records_and_aggregates_accuracy_and_f1(monkeypatch):
    monkeypatch.setattr(node_main.args, "epoch", 1)
    monkeypatch.setattr(node_main.args, "tst_epoch", 1)
    monkeypatch.setattr(node_main.args, "assignment", "top1")

    experiment = node_main.Exp(_MultiHandler())
    experiment.prepare_model = lambda: None
    experiment.train_epoch = lambda: {"Loss": 0.0, "preLoss": 0.0}
    experiment.save_history = lambda: None
    experiment.model = _Model()
    experiment.test_epoch = lambda handler, dataset_id: {
        "Acc": 0.75,
        "F1": 0.5,
        "tstNum": 4,
    }

    experiment.run()

    assert experiment.metrics["Test0Acc"] == [0.75]
    assert experiment.metrics["Test0F1"] == [0.5]


def test_resumed_training_adds_classification_metrics_to_loaded_history(monkeypatch):
    monkeypatch.setattr(node_main.args, "epoch", 1)
    monkeypatch.setattr(node_main.args, "tst_epoch", 1)
    monkeypatch.setattr(node_main.args, "assignment", "top1")
    monkeypatch.setattr(node_main.args, "load_model", "pretrained")

    experiment = node_main.Exp(_MultiHandler())
    experiment.prepare_model = lambda: None
    experiment.train_epoch = lambda: {"Loss": 0.0, "preLoss": 0.0}
    experiment.save_history = lambda: None
    experiment.model = _Model()
    experiment.load_model = lambda: setattr(
        experiment,
        "metrics",
        {
            "TrainLoss": [],
            "TrainpreLoss": [],
            "Test0Recall": [],
            "Test0NDCG": [],
        },
    )
    experiment.test_epoch = lambda handler, dataset_id: {
        "Acc": 0.75,
        "F1": 0.5,
        "tstNum": 4,
    }

    experiment.run()

    assert experiment.metrics["Test0Acc"] == [0.75]
    assert experiment.metrics["Test0F1"] == [0.5]
