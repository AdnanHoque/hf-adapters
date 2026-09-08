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


def load(enabled):
    names = {"_prefill_expert_config", "_moe_expert_persistent"}
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


def test_opt_in_requests_each_measured_matmul_division():
    observed = []
    original = torch.matmul
    with hints() as active:

        def multiply(a, b):
            division = {}
            for hint in active:
                division.update(hint.get("work_div", {}))
            observed.append(division)
            return original(a, b)

        with patch.object(torch, "matmul", multiply):
            result = load(True)["_moe_expert_persistent"](*args())
    assert tuple(result.shape) == (512, 2816)
    assert observed == [{"T": 8, "H": 4}, {"T": 8, "H": 4}, {"T": 16, "H": 2}]


@pytest.mark.parametrize(
    "overrides",
    [
        {"experts": 2},
        {"tokens": 256},
        {"hidden": 1408},
        {"width": 768},
        {"dtype": torch.bfloat16},
    ],
)
def test_explicit_unsupported_request_declines(overrides):
    with hints(), pytest.raises(ValueError, match="FP16 E128"):
        load(True)["_moe_expert_persistent"](*args(**overrides))


def compiler_config():
    return SimpleNamespace(
        consumer_compatible_input_staging=False,
        read_copy_elision=False,
        sencores=32,
        layout_solver="greedy",
        co_optimizing_lx_planning=False,
        ktir_emitter=False,
        ignore_work_division_hints=False,
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
        ("lx_planning", False),
    ],
)
def test_incompatible_compiler_request_is_explicit(name, value):
    config = compiler_config()
    setattr(config, name, value)
    with with_config(config), pytest.raises(RuntimeError, match="32 cores"):
        load(True)["_prefill_expert_config"]()


@pytest.mark.parametrize(
    "name", ["consumer_compatible_input_staging", "read_copy_elision"]
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
