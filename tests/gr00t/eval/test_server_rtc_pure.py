"""Capability handshake is tied to the loaded action head, not a CLI claim."""

from types import SimpleNamespace

import pytest
from tests.gr00t.eval import test_server_model_identity as server_test_support


server_functions = server_test_support.server_functions


@pytest.mark.parametrize("support", [True, False, None])
def test_server_advertises_actual_pure_sampler_support(server_functions, support):
    captured = {}

    class FakeServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def run(self):
            pass

    server_functions.update(
        EmbodimentTag=SimpleNamespace(
            resolve=lambda value: SimpleNamespace(value=value, name="TEST_ROBOT")
        ),
        Gr00tPolicy=lambda **kwargs: SimpleNamespace(
            get_vision_input_contract=lambda: {},
            model=SimpleNamespace(action_head=SimpleNamespace(supports_rtc_pure=support)),
        ),
        PolicyServer=FakeServer,
    )
    server_functions["main"](
        SimpleNamespace(
            model_path="outputs/existing/checkpoint-30000",
            embodiment_tag="test_robot",
            device="cpu",
            host="unused",
            port=5555,
            strict=True,
            use_sim_policy_wrapper=False,
            deployment_dataset_path=None,
        )
    )
    rtc = captured["policy_metadata"]["rtc"]
    assert rtc["pure_inference_guidance"] is (support is True)
    assert rtc["physical_action_tail"] is True
    assert rtc["protocol_version"] == 1
