import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path(__file__).resolve().parents[1]
MODEL_PATHS = [REPO / "model.py", REPO / "node_classification" / "model.py"]


def _load_model(path, index):
    params = types.ModuleType("params")
    params.args = SimpleNamespace(devices=["cuda:0", "cuda:1"])
    logger = types.ModuleType("Utils.TimeLogger")
    logger.log = lambda *args, **kwargs: None
    utils = types.ModuleType("Utils")
    utils.__path__ = []
    utils.TimeLogger = logger
    previous = {name: sys.modules.get(name) for name in ("params", "Utils", "Utils.TimeLogger")}
    sys.modules.update({"params": params, "Utils": utils, "Utils.TimeLogger": logger})
    try:
        spec = importlib.util.spec_from_file_location(f"anygraph_model_{index}", path)
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


def test_attempt_sampling_indices_follow_rows_device_in_both_models():
    rows = torch.arange(8)
    for index, path in enumerate(MODEL_PATHS):
        module, previous = _load_model(path, index)
        try:
            indices = module._attempt_sample_indices(rows, 3)
            assert indices.device == rows.device
            assert indices.numel() == 3
            assert indices.unique().numel() == 3
        finally:
            _restore_modules(previous)
