# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only recording identity coverage, without importing model/GPU dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def server_functions():
    # Compile only the pure helper and entrypoint; injected doubles below ensure
    # the metadata wiring cannot load weights, bind a socket, or run inference.
    source = Path(__file__).resolve().parents[3] / "gr00t/eval/run_gr00t_server.py"
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_model_identity", "main"}
    ]
    namespace = {"Path": Path, "ServerConfig": SimpleNamespace}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("model_path", "model_name", "checkpoint"),
    [
        ("/models/depth_run/checkpoint-10000", "depth_run", "checkpoint-10000"),
        ("outputs/rgb_run/checkpoint-4200/", "rgb_run", "checkpoint-4200"),
        ("nvidia/GR00T-N1.7-3B", "GR00T-N1.7-3B", None),
        ("/models/final-model", "final-model", None),
        ("/models/checkpoint-experiment", "checkpoint-experiment", None),
    ],
)
def test_model_identity_describes_configured_model(
    server_functions, model_path, model_name, checkpoint
):
    assert server_functions["_model_identity"](model_path) == {
        "model_name": model_name,
        "checkpoint": checkpoint,
        "model_path": model_path,
    }


def test_relative_checkpoint_uses_current_model_directory(server_functions, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert server_functions["_model_identity"]("checkpoint-4200") == {
        "model_name": tmp_path.name,
        "checkpoint": "checkpoint-4200",
        "model_path": "checkpoint-4200",
    }


def test_model_identity_is_passed_to_policy_server(server_functions):
    captured = {}

    class FakeServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def run(self):
            pass

    server_functions.update(
        EmbodimentTag=SimpleNamespace(
            resolve=lambda value: SimpleNamespace(value=value, name="TEST_ROBOT")
        ),
        Gr00tPolicy=lambda **kwargs: SimpleNamespace(get_vision_input_contract=lambda: {}),
        PolicyServer=FakeServer,
    )
    config = SimpleNamespace(
        model_path="outputs/depth_run/checkpoint-10000",
        embodiment_tag="test_robot",
        device="cpu",
        host="unused",
        port=5555,
        strict=True,
        use_sim_policy_wrapper=False,
        deployment_dataset_path=None,
    )

    server_functions["main"](config)

    assert captured["policy_metadata"]["model_identity"] == {
        "model_name": "depth_run",
        "checkpoint": "checkpoint-10000",
        "model_path": config.model_path,
    }
