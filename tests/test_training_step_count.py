import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
MAIN_PATHS = [REPO / "main.py", REPO / "node_classification" / "main.py"]


def _load_main(path, index):
    names = ["params", "model", "data_handler", "setproctitle", "Utils", "Utils.TimeLogger"]
    previous = {name: sys.modules.get(name) for name in names}
    params = types.ModuleType("params")
    params.args = SimpleNamespace(epoch_max_step=-1)
    model = types.ModuleType("model")
    for name in ("Expert", "Feat_Projector", "Adj_Projector", "AnyGraph"):
        setattr(model, name, object)
    data_handler = types.ModuleType("data_handler")
    data_handler.MultiDataHandler = object
    data_handler.DataHandler = object
    setproctitle = types.ModuleType("setproctitle")
    logger = types.ModuleType("Utils.TimeLogger")
    logger.log = lambda *args, **kwargs: None
    logger.saveDefault = False
    utils = types.ModuleType("Utils")
    utils.__path__ = []
    utils.TimeLogger = logger
    sys.modules.update({
        "params": params,
        "model": model,
        "data_handler": data_handler,
        "setproctitle": setproctitle,
        "Utils": utils,
        "Utils.TimeLogger": logger,
    })
    try:
        spec = importlib.util.spec_from_file_location(f"anygraph_main_steps_{index}", path)
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


def test_epoch_max_step_sets_the_regularizer_denominator_in_both_modes():
    for index, path in enumerate(MAIN_PATHS):
        module, previous = _load_main(path, index)
        try:
            assert module._training_step_count(100, -1) == 100
            assert module._training_step_count(100, 0) == 100
            assert module._training_step_count(100, 7) == 7
            assert module._training_step_count(5, 7) == 5
            assert module._training_step_count(0, 7) == 0
        finally:
            _restore_modules(previous)
