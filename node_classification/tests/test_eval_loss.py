import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


NODE_DIR = Path(__file__).resolve().parents[1]


def test_eval_loss_cli_parses_false_value():
    spec = importlib.util.spec_from_file_location("node_params_eval_loss_test", NODE_DIR / "params.py")
    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = ["params.py", "--eval_loss", "False"]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    assert module.args.eval_loss is False
    assert module.parse_bool("yes") is True
    assert module.parse_bool("0") is False


def _load_exp(eval_loss):
    names = ["main", "model", "params", "data_handler", "setproctitle", "Utils", "Utils.TimeLogger"]
    previous = {name: sys.modules.get(name) for name in names}
    node_dir_str = str(NODE_DIR)
    sys.path.insert(0, node_dir_str)

    logger = types.ModuleType("Utils.TimeLogger")
    logger.log = lambda *args, **kwargs: None
    logger.saveDefault = False
    utils = types.ModuleType("Utils")
    utils.__path__ = []
    utils.TimeLogger = logger
    params = types.ModuleType("params")
    params.args = SimpleNamespace(eval_loss=eval_loss)
    data_handler = types.ModuleType("data_handler")
    data_handler.MultiDataHandler = object
    data_handler.DataHandler = object
    setproctitle = types.ModuleType("setproctitle")
    setproctitle.setproctitle = lambda *_: None

    sys.modules.update({
        "Utils": utils,
        "Utils.TimeLogger": logger,
        "params": params,
        "data_handler": data_handler,
        "setproctitle": setproctitle,
    })
    try:
        model = importlib.import_module("model")
        main = importlib.import_module("main")
    except Exception:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.path.remove(node_dir_str)
        raise
    return main.Exp, model.Expert, previous, node_dir_str


def _restore_modules(previous, node_dir_str):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    sys.path.remove(node_dir_str)


class _Loader:
    def __init__(self, batches):
        self.batches = batches
        self.dataset = SimpleNamespace(class_num=2)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _Handler:
    data_name = "synthetic"
    projectors = None

    def __init__(self):
        self.tst_loader = _Loader([
            (torch.tensor([0]), torch.tensor([0])),
            (torch.tensor([1]), torch.tensor([1])),
        ])


def test_test_epoch_reports_mean_cross_entropy_from_synthetic_logits():
    Exp, Expert, previous, node_dir_str = _load_exp(eval_loss=True)
    try:
        class SyntheticExpert(Expert):
            def __init__(self):
                nn.Module.__init__(self)
                self.embeddings = torch.tensor([
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ])

            def forward(self, _feats):
                return self.embeddings

        expert = SyntheticExpert()

        class Model:
            def eval(self):
                pass

            def summon(self, _dataset_id):
                return expert

        exp = Exp.__new__(Exp)
        exp.model = Model()
        handler = _Handler()
        original_cuda = torch.Tensor.cuda
        try:
            torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
            result = exp.test_epoch(handler, 0)
        finally:
            torch.Tensor.cuda = original_cuda

        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1])
        ).item()
        assert result["Acc"] == 1.0
        assert result["F1"] == 1.0
        assert abs(result["Loss"] - expected) < 1e-7
        assert result["tstNum"] == 2
    finally:
        _restore_modules(previous, node_dir_str)


def test_test_epoch_omits_cross_entropy_when_disabled():
    Exp, Expert, previous, node_dir_str = _load_exp(eval_loss=False)
    try:
        class SyntheticExpert(Expert):
            def __init__(self):
                nn.Module.__init__(self)
                self.embeddings = torch.tensor([
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ])

            def forward(self, _feats):
                return self.embeddings

        expert = SyntheticExpert()

        class Model:
            def eval(self):
                pass

            def summon(self, _dataset_id):
                return expert

        exp = Exp.__new__(Exp)
        exp.model = Model()
        original_cuda = torch.Tensor.cuda
        try:
            torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
            result = exp.test_epoch(_Handler(), 0)
        finally:
            torch.Tensor.cuda = original_cuda

        assert result["Acc"] == 1.0
        assert "Loss" not in result
    finally:
        _restore_modules(previous, node_dir_str)
