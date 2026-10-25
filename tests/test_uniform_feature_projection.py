import sys

import pytest
import torch


# params.py uses argparse at import time; keep pytest's CLI flags out of its parser.
_original_argv = sys.argv
sys.argv = [_original_argv[0]]
try:
    import model as anygraph_model
    from node_classification import model as node_classification_model
    from params import args
finally:
    sys.argv = _original_argv


@pytest.mark.parametrize(
    "model_module",
    [anygraph_model, node_classification_model],
    ids=["link-prediction", "node-classification"],
)
def test_uniform_projection_uses_the_input_feature_width(model_module):
    original_method = args.proj_method
    original_latdim = args.latdim
    had_global_featdim = hasattr(args, "featdim")
    original_global_featdim = getattr(args, "featdim", None)
    try:
        args.proj_method = "uniform"
        # Simulate a different feature-bearing dataset being loaded last.
        args.featdim = 7
        args.latdim = 4

        projected = model_module.Feat_Projector(torch.ones((3, 2))).forward()

        assert projected.shape == (3, 4)
    finally:
        args.proj_method = original_method
        args.latdim = original_latdim
        if had_global_featdim:
            args.featdim = original_global_featdim
        else:
            del args.featdim
