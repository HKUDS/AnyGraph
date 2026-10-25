import sys
from pathlib import Path

import pytest

NODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE_DIR))

_original_argv = sys.argv
sys.argv = [_original_argv[0], "--dataset", "node"]
try:
    from params import parse_args
finally:
    sys.argv = _original_argv


def _parse(argv):
    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        return parse_args()
    finally:
        sys.argv = original_argv


def test_dataset_setting_must_be_selected_instead_of_invalid_default():
    with pytest.raises(SystemExit) as error:
        _parse([])

    assert error.value.code == 2


def test_readme_dataset_alias_selects_node_group():
    assert _parse(["--dataset", "node"]).dataset_setting == "node"
