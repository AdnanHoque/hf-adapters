# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test shipped functions without optional model imports; no device claims."""

import ast
import contextlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "hf_adapters/hf_gemma4_moe.py"


def load(enabled=None):
    names = {
        "_prefill_expert_config",
        "_validate_prefill_expert_inputs",
        "_moe_expert_persistent",
    }
    nodes = [
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(nodes) == len(names)
    ns = {
        "torch": torch,
        "F": torch.nn.functional,
        "nullcontext": contextlib.nullcontext,
        "_PREFILL_EXPERT_DIVISIONS": enabled,
    }
    exec(compile(ast.Module(nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


@contextlib.contextmanager
def hints():
    active = []

    @contextlib.contextmanager
    def hint(**kwargs):
        active.append(kwargs)
        try:
            yield
        finally:
            active.pop()

    module = ModuleType("torch_spyre._inductor.propagate_hints")
    module.spyre_hint = hint
    with patch.dict(sys.modules, {module.__name__: module}):
        yield active


def args(
    experts=128, tokens=512, hidden=2816, width=704, dtype=torch.float16, device="meta"
):
    return [
        torch.empty(shape, dtype=dtype, device=device)
        for shape in [
            (tokens, hidden),
            (tokens, experts, 1),
            (experts, hidden, width),
            (experts, hidden, width),
            (experts, width, hidden),
        ]
    ]


def test_default_arithmetic_and_hint_are_unchanged():
    torch.manual_seed(0)
    x, route, gate, up, down = [
        torch.randn_like(t)
        for t in args(experts=2, tokens=4, hidden=8, width=4, device="cpu")
    ]
    with hints():
        actual = load(False)["_moe_expert_persistent"](x, route, gate, up, down)
    expected = (
        (
            torch.nn.functional.gelu(x.unsqueeze(0) @ gate, approximate="tanh")
            * (x.unsqueeze(0) @ up)
        )
        @ down
        * route.permute(1, 0, 2)
    ).sum(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert load(False)["_prefill_expert_config"]() == {
        "allow_all_ops_in_lx_planning": True
    }


def test_source_default_is_automatic():
    flag = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_PREFILL_EXPERT_DIVISIONS"
            for t in n.targets
        )
    )
    assert ast.literal_eval(flag.value) is None


@pytest.mark.parametrize(
    "name",
    [
        "consumer_compatible_input_staging",
        "read_copy_elision",
        "lx_planner_relayout",
    ],
)
def test_default_missing_capability_uses_ordinary_hints(name):
    config = compiler_config()
    delattr(config, name)
    observed = []
    original = torch.matmul
    with with_config(config), hints() as active:
        assert load()["_prefill_expert_config"]() == {
            "allow_all_ops_in_lx_planning": True
        }

        def multiply(a, b):
            division = {}
            for hint in active:
                division.update(hint.get("work_div", {}))
            observed.append(division)
            return original(a, b)

        with patch.object(torch, "matmul", multiply):
            load()["_moe_expert_persistent"](*args())
    assert observed == [{"T": 32}] * 3


@pytest.mark.parametrize(
    "name,value",
    [
        ("sencores", 16),
        ("layout_solver", "cpsat"),
        ("co_optimizing_lx_planning", True),
        ("ktir_emitter", True),
        ("ignore_work_division_hints", True),
        ("ignore_wsr_hints", True),
        ("lx_planning", False),
    ],
)
def test_default_incompatible_config_uses_ordinary_path(name, value):
    config = compiler_config()
    setattr(config, name, value)
    with with_config(config):
        assert load()["_prefill_expert_config"]() == {
            "allow_all_ops_in_lx_planning": True
        }


@pytest.mark.parametrize(
    "overrides",
    [
        {"experts": 2},
        {"tokens": 256},
        {"hidden": 1408},
        {"width": 768},
        {"dtype": torch.float32},
    ],
)
def test_default_unsupported_shape_keeps_ordinary_arithmetic(overrides):
    inputs = args(**overrides)
    with hints(), patch.object(torch, "matmul", wraps=torch.matmul) as multiply:
        result = load()["_moe_expert_persistent"](*inputs)
    assert tuple(result.shape) == tuple(inputs[0].shape)
    assert multiply.call_count == 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_automatic_default_requests_each_measured_matmul_division(dtype):
    observed = []
    original = torch.matmul
    with with_config(compiler_config()), hints() as active:

        def multiply(a, b):
            division = {}
            for hint in active:
                division.update(hint.get("work_div", {}))
            observed.append(division)
            return original(a, b)

        with patch.object(torch, "matmul", multiply):
            result = load()["_moe_expert_persistent"](*args(dtype=dtype))
    assert tuple(result.shape) == (512, 2816)
    assert observed == [{"T": 8, "H": 4}, {"T": 8, "H": 4}, {"T": 16, "H": 2}]


@pytest.mark.parametrize(
    "overrides",
    [
        {"experts": 2},
        {"tokens": 256},
        {"hidden": 1408},
        {"width": 768},
        {"dtype": torch.float32},
    ],
)
def test_explicit_unsupported_request_declines(overrides):
    with hints(), pytest.raises(ValueError, match="FP16/BF16 E128"):
        load(True)["_moe_expert_persistent"](*args(**overrides))


@pytest.mark.parametrize("input_index", [1, 2, 3, 4])
def test_mixed_16_bit_inputs_decline(input_index):
    inputs = args(dtype=torch.bfloat16)
    inputs[input_index] = inputs[input_index].to(torch.float16)
    x, route, gate, up, down = inputs
    assert not load()["_validate_prefill_expert_inputs"](x, gate, up, down, route)
    with pytest.raises(ValueError, match="FP16/BF16 E128"):
        load(True)["_validate_prefill_expert_inputs"](x, gate, up, down, route)


def compiler_config():
    return SimpleNamespace(
        consumer_compatible_input_staging=False,
        read_copy_elision=False,
        lx_planner_relayout=False,
        sencores=32,
        layout_solver="greedy",
        co_optimizing_lx_planning=False,
        ktir_emitter=False,
        ignore_work_division_hints=False,
        ignore_wsr_hints=False,
        lx_planning=True,
    )


def with_config(config):
    module = ModuleType("torch_spyre._inductor")
    module.config = config
    return patch.dict(sys.modules, {module.__name__: module})


def test_opt_in_returns_explicit_capabilities_without_mutating_globals():
    config = compiler_config()
    before = vars(config).copy()
    with with_config(config):
        assert load(True)["_prefill_expert_config"]() == {
            "allow_all_ops_in_lx_planning": True,
            "lx_planner_relayout": True,
            "consumer_compatible_input_staging": True,
            "read_copy_elision": True,
        }
    assert vars(config) == before


@pytest.mark.parametrize(
    "name,value",
    [
        ("sencores", 16),
        ("layout_solver", "cpsat"),
        ("co_optimizing_lx_planning", True),
        ("ktir_emitter", True),
        ("ignore_work_division_hints", True),
        ("ignore_wsr_hints", True),
        ("lx_planning", False),
    ],
)
def test_incompatible_compiler_request_is_explicit(name, value):
    config = compiler_config()
    setattr(config, name, value)
    with with_config(config), pytest.raises(RuntimeError, match="32 cores"):
        load(True)["_prefill_expert_config"]()


@pytest.mark.parametrize(
    "name",
    ["consumer_compatible_input_staging", "read_copy_elision", "lx_planner_relayout"],
)
def test_missing_capability_is_explicit(name):
    config = compiler_config()
    delattr(config, name)
    with with_config(config), pytest.raises(RuntimeError, match="require"):
        load(True)["_prefill_expert_config"]()


def test_existing_prefill_boundary_scopes_and_cleans_up_on_error():
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Gemma4MoEBlock"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    events = []

    @contextlib.contextmanager
    def scope(options):
        events.append(("enter", options))
        try:
            yield
        finally:
            events.append("exit")

    def fail(*a):
        raise RuntimeError("compile failed")

    ns = {
        "_prefill_expert_config": lambda: {"sentinel": True},
        "_validate_prefill_expert_inputs": lambda *a: True,
        "_name_prefill_inputs": lambda *a: events.append("name"),
        "_reset_named_dims": lambda: events.append("reset"),
        "optional_spyre_config_patch": scope,
    }
    exec(compile(ast.Module([forward], type_ignores=[]), str(SOURCE), "exec"), ns)
    instance = SimpleNamespace(
        experts=SimpleNamespace(gate_proj=None, up_proj=None, down_proj=None),
        _compiled_prefill_attn=lambda *a: (a[0], a[3], a[4]),
        _compiled_prefill_ffn=fail,
    )
    with pytest.raises(RuntimeError, match="compile failed"):
        ns["forward"](
            instance, torch.empty(1, 512, 1), None, None, None, None, None, None
        )
    assert events == ["name", ("enter", {"sentinel": True}), "exit", "reset"]


def forward_with(ns):
    cls = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Gemma4MoEBlock"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    exec(compile(ast.Module([forward], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["forward"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"tokens": 256},
        {"hidden": 1408},
        {"width": 768},
        {"experts": 64},
        {"dtype": torch.float32},
    ],
)
def test_shape_errors_precede_attention_and_cache_writes(overrides):
    x, _, gate, up, down = args(**overrides)
    key, value = torch.zeros(1), torch.zeros(1)
    events = []

    def attention(*a):
        events.append("attention")
        key.add_(1)
        value.add_(1)
        return a[0], key, value

    instance = SimpleNamespace(
        experts=SimpleNamespace(gate_proj=gate, up_proj=up, down_proj=down),
        _compiled_prefill_attn=attention,
    )
    with (
        with_config(compiler_config()),
        pytest.raises(ValueError, match="FP16/BF16 E128"),
    ):
        forward_with(load(True))(
            instance, x.unsqueeze(0), None, None, key, value, None, None
        )
    assert events == []
    assert key.item() == value.item() == 0


def test_loop_hint_disabled_fails_before_attention():
    config = compiler_config()
    config.ignore_wsr_hints = True
    with with_config(config), pytest.raises(RuntimeError, match="honored hints"):
        # No attention/experts attributes: either access would fail this test.
        forward_with(load(True))(
            SimpleNamespace(),
            torch.empty(1, 512, 2816, device="meta"),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def test_partial_dimension_naming_is_cleaned_up():
    x, _, gate, up, down = args()
    events = []

    def fail_naming(*a):
        events.append("partial name")
        raise RuntimeError("naming failed")

    ns = load(True)
    ns.update(
        _name_prefill_inputs=fail_naming,
        _reset_named_dims=lambda: events.append("reset"),
    )
    instance = SimpleNamespace(
        experts=SimpleNamespace(gate_proj=gate, up_proj=up, down_proj=down),
        _compiled_prefill_attn=lambda *a: (a[0], a[3], a[4]),
    )
    with (
        with_config(compiler_config()),
        pytest.raises(RuntimeError, match="naming failed"),
    ):
        forward_with(ns)(instance, x.unsqueeze(0), None, None, None, None, None, None)
    assert events == ["partial name", "reset"]


@pytest.mark.parametrize(
    "bad_route",
    [
        torch.empty(256, 128, 1, device="meta", dtype=torch.float16),
        torch.empty(512, 128, 1, device="meta", dtype=torch.float32),
    ],
)
def test_region_retains_routing_guard(bad_route):
    inputs = args()
    inputs[1] = bad_route
    with hints(), pytest.raises(ValueError, match="FP16/BF16 E128"):
        load(True)["_moe_expert_persistent"](*inputs)


def test_region_requires_flat_rows_even_though_forward_accepts_batched_input():
    inputs = args()
    inputs[0] = inputs[0].unsqueeze(0)
    with hints(), pytest.raises(ValueError, match="FP16/BF16 E128"):
        load(True)["_moe_expert_persistent"](*inputs)


def test_decode_does_not_request_prefill_capabilities():
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Gemma4MoEBlock"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )

    def forbidden():
        raise AssertionError("decode entered prefill configuration")

    ns = {"_prefill_expert_config": forbidden}
    exec(compile(ast.Module([forward], type_ignores=[]), str(SOURCE), "exec"), ns)
    instance = SimpleNamespace(_compiled_decode=lambda *a: (a[0], "k", "v"))
    x = torch.empty(1, 1, 1)
    assert ns["forward"](instance, x, None, None, None, None, None, None) == (
        x,
        "k",
        "v",
    )
