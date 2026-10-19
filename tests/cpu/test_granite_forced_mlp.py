# Copyright 2025 The Torch-Spyre Authors.
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

"""The opt-in selects only the checked model, rank group and prefill shape."""

from types import SimpleNamespace

import torch
from transformers.models.granite.modeling_granite import GraniteMLP

from hf_adapters.hf_common import PaddedGatedMLP
from hf_adapters.hf_granite import (
    _can_force_mlp,
    _GraniteTP4PrefillMLP,
    _prepare_mlp,
)
from hf_adapters.spyre_tensor_parallel import SpyreColwiseParallel, SpyreRowwiseParallel


def test_granite_forced_mlp_option_and_guards(monkeypatch):
    config = SimpleNamespace(
        model_type="granite",
        hidden_size=4096,
        intermediate_size=12800,
        num_hidden_layers=40,
        hidden_act="silu",
        mlp_bias=False,
    )
    local = SimpleNamespace(**vars(config))
    local.intermediate_size = 3200
    with torch.device("meta"):
        mlp = GraniteMLP(local).to(dtype=torch.float16).eval()
    mesh = SimpleNamespace(ndim=1, size=lambda: 4)
    for name, style in (
        ("gate_proj", SpyreColwiseParallel()),
        ("up_proj", SpyreColwiseParallel()),
        ("down_proj", SpyreRowwiseParallel()),
    ):
        style.prepare_module_tp(getattr(mlp, name), mesh)
    monkeypatch.delenv("SPYRE_GRANITE_TP4_PREFILL_MLP", raising=False)
    assert not _can_force_mlp(mlp, config)
    monkeypatch.setenv("SPYRE_GRANITE_TP4_PREFILL_MLP", "1")
    assert _can_force_mlp(mlp, config)
    for field, value in (
        ("model_type", "llama"),
        ("hidden_size", 2048),
        ("intermediate_size", 14336),
        ("num_hidden_layers", 32),
    ):
        other = SimpleNamespace(**vars(config))
        setattr(other, field, value)
        assert not _can_force_mlp(mlp, other)
    mesh.size = lambda: 1
    assert not _can_force_mlp(mlp, config)
    mesh.size = lambda: 4
    mlp.up_proj._spyre_mlp_width_mesh = None
    assert not _can_force_mlp(mlp, config)
    mlp.up_proj._spyre_mlp_width_mesh = mesh
    mlp.bfloat16()
    assert not _can_force_mlp(mlp, config)
    mlp.half()

    # Allocate only for the actual preparation; padding must retain TP hooks.
    mlp.to_empty(device="cpu")
    down = mlp.down_proj
    prepared = _prepare_mlp(mlp, config)
    assert type(prepared) is _GraniteTP4PrefillMLP
    assert prepared.down_proj is down
    assert prepared.gate_proj.weight.shape == (3328, 4096)
    assert prepared.down_proj.weight.shape == (4096, 3200)
    assert prepared.gate_proj._forward_hooks == mlp.gate_proj._forward_hooks
    x = SimpleNamespace(
        device=SimpleNamespace(type="spyre"), dtype=torch.float16, shape=(1, 512, 4096)
    )
    assert prepared._use_forced_division(x)
    for shape in ((1, 1, 4096), (2, 512, 4096), (1, 256, 4096), (512, 4096)):
        x.shape = shape
        assert not prepared._use_forced_division(x)
    x.shape = (1, 512, 4096)
    x.device.type = "cpu"
    assert not prepared._use_forced_division(x)
    x.device.type = "spyre"
    x.dtype = torch.bfloat16
    assert not prepared._use_forced_division(x)
    x.dtype = torch.float16
    prepared.train()
    assert not prepared._use_forced_division(x)

    # The default still uses the existing padded module.
    monkeypatch.delenv("SPYRE_GRANITE_TP4_PREFILL_MLP")
    tiny = SimpleNamespace(
        hidden_size=16, intermediate_size=20, hidden_act="silu", mlp_bias=False
    )
    default = _prepare_mlp(GraniteMLP(tiny).half().eval(), config)
    assert type(default) is PaddedGatedMLP
