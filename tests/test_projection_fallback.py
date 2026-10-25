import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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
    import data_handler as root_data_handler
finally:
    sys.argv = _original_argv


class _Projector:
    def __init__(self, output):
        self.output = output

    def __call__(self):
        return self.output


class _FailOnceTopoEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, adjacency, features, normed=False):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("synthetic first-pass memory failure")
        return features


@pytest.mark.parametrize("feature_width", [100, 300, 512])
def test_root_memory_fallback_preserves_all_feature_columns(monkeypatch, feature_width):
    args = root_data_handler.args
    monkeypatch.setattr(args, "proj_method", "both")
    monkeypatch.setattr(args, "devices", ["cpu"], raising=False)
    monkeypatch.setattr(args, "tst_mode", "tst")
    monkeypatch.setattr(args, "trn_mode", "train-all")

    combined_features = torch.arange(2 * feature_width, dtype=torch.float32).reshape(
        2, feature_width
    )
    monkeypatch.setattr(
        root_data_handler,
        "Adj_Projector",
        lambda adjacency: _Projector(torch.zeros_like(combined_features)),
    )
    monkeypatch.setattr(
        root_data_handler,
        "Feat_Projector",
        lambda features: _Projector(combined_features),
    )

    handler = object.__new__(root_data_handler.DataHandler)
    handler.data_name = "synthetic"
    handler.asym_adj = torch.eye(2).to_sparse()
    handler.trn_input_adj = torch.eye(2).to_sparse()
    handler.feats = torch.ones((2, feature_width))
    handler.topo_encoder = _FailOnceTopoEncoder()

    handler.make_projectors()

    assert handler.projectors.shape == (2, feature_width)
    assert torch.isfinite(handler.projectors).all()
