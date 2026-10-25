import argparse
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
    from params import parse_tst_steps
finally:
    sys.argv = _original_argv


class _Loader(list):
    dataset = SimpleNamespace(class_num=2)


class _Expert:
    def pred_for_node_test(self, nodes, class_num, features, rerun_embed=True):
        return nodes % class_num


class _Model:
    def eval(self):
        pass

    def summon(self, dataset_id):
        return _Expert()


def test_node_classification_test_steps_limit_batches(monkeypatch):
    monkeypatch.setattr(node_main.args, "tst_steps", 1)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)

    experiment = object.__new__(node_main.Exp)
    experiment.model = _Model()
    handler = SimpleNamespace(
        tst_loader=_Loader(
            [
                (torch.tensor([0, 1]), torch.tensor([0, 1])),
                (torch.tensor([2, 3]), torch.tensor([0, 0])),
            ]
        ),
        projectors=torch.empty(0),
    )

    result = experiment.test_epoch(handler, 0)

    assert result["tstNum"] == 2
    assert result["Acc"] == 1.0
    assert result["F1"] == 1.0


@pytest.mark.parametrize("value", ["0", "-2", "many"])
def test_test_steps_reject_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_tst_steps(value)


def test_node_classification_test_steps_minus_one_keeps_all_batches(monkeypatch):
    monkeypatch.setattr(node_main.args, "tst_steps", -1)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)

    experiment = object.__new__(node_main.Exp)
    experiment.model = _Model()
    handler = SimpleNamespace(
        tst_loader=_Loader(
            [
                (torch.tensor([0, 1]), torch.tensor([0, 1])),
                (torch.tensor([2, 3]), torch.tensor([0, 0])),
            ]
        ),
        projectors=torch.empty(0),
    )

    result = experiment.test_epoch(handler, 0)

    assert result["tstNum"] == 4
    assert result["Acc"] == 0.75
