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

"""Load shipped functions for dependency-free CPU geometry tests.

These tests stub only Spyre hints/model imports, not tensor operations. They
do not substitute for compiled layout, placement or numerical device tests.
"""

import ast
import contextlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[2] / "hf_adapters/hf_gemma4_moe.py"


def load_functions(names, **overrides):
    parsed = ast.parse(SOURCE.read_text())
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "nullcontext": contextlib.nullcontext,
    }
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id.startswith("_DECODE_"):
                    namespace[name.id] = ast.literal_eval(node.value)
    namespace.update(overrides)
    nodes = [
        n
        for n in ast.walk(parsed)
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if len(nodes) != len(names):
        raise AssertionError(f"Expected exactly the requested functions: {names}")
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return namespace


@contextlib.contextmanager
def recorded_hints():
    calls = []

    @contextlib.contextmanager
    def spyre_hint(**kwargs):
        calls.append(kwargs)
        yield

    module = ModuleType("torch_spyre._inductor.propagate_hints")
    module.spyre_hint = spyre_hint
    with patch.dict(sys.modules, {module.__name__: module}):
        yield calls
