import importlib.util
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
HANDLER_PATHS = [REPO / "data_handler.py", REPO / "node_classification" / "data_handler.py"]


def _load_handler(path, index):
    names = ["params", "model", "Utils", "Utils.TimeLogger", "torch_geometric", "torch_geometric.transforms"]
    previous = {name: sys.modules.get(name) for name in names}
    params = types.ModuleType("params")
    params.args = SimpleNamespace()
    logger = types.ModuleType("Utils.TimeLogger")
    logger.log = lambda *args, **kwargs: None
    utils = types.ModuleType("Utils")
    utils.__path__ = []
    utils.TimeLogger = logger
    model = types.ModuleType("model")
    for name in ("Feat_Projector", "Adj_Projector", "TopoEncoder"):
        setattr(model, name, object)
    torch_geometric = types.ModuleType("torch_geometric")
    torch_geometric.__path__ = []
    transforms = types.ModuleType("torch_geometric.transforms")
    torch_geometric.transforms = transforms
    sys.modules.update({
        "params": params,
        "Utils": utils,
        "Utils.TimeLogger": logger,
        "model": model,
        "torch_geometric": torch_geometric,
        "torch_geometric.transforms": transforms,
    })
    try:
        spec = importlib.util.spec_from_file_location(f"anygraph_data_handler_{index}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, previous
    except Exception:
        _restore_modules(previous)
        raise


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.mark.parametrize("handler_index", [0, 1])
def test_corrupt_feature_file_raises_contextual_error_not_success_exit(tmp_path, handler_index):
    module, previous = _load_handler(HANDLER_PATHS[handler_index], handler_index)
    try:
        feature_path = tmp_path / "corrupt-features.pkl"
        feature_path.write_bytes(b"not a valid pickle")
        handler = module.DataHandler.__new__(module.DataHandler)

        with pytest.raises(RuntimeError, match="Failed to load features") as exc_info:
            handler.load_feats(str(feature_path))

        assert isinstance(exc_info.value.__cause__, pickle.UnpicklingError)
    finally:
        _restore_modules(previous)
