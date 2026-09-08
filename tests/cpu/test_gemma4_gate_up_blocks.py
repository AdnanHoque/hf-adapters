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

"""Exact small-integer geometry tests, not a device arithmetic tolerance."""

import ast
import unittest
from unittest.mock import patch

import torch
from _gemma4_decode_perf_helpers import SOURCE, load_functions, recorded_hints


class GateUpBlockTests(unittest.TestCase):
    def test_partial_sums_cover_inputs_and_keep_sequential_panel_use(self):
        namespace = load_functions({"_decode_gate_up_blocks"})
        gate = (torch.arange(8 * 11 * 3).reshape(8, 11, 3) % 7).double()
        up = gate + 2
        inputs = (torch.arange(8 * 11).reshape(8, 1, 11) % 5).double() / 8
        ids = torch.tensor([[0, 0, 7, 7, 3, 6, 1, 2]])
        bmm = torch.bmm
        calls = []

        def record_bmm(x, y):
            calls.append((x.clone(), y.clone()))
            return bmm(x, y)

        with recorded_hints() as hints, patch.object(torch, "bmm", record_bmm):
            actual_gate, actual_up = namespace["_decode_gate_up_blocks"](
                inputs, gate, up, ids, 4
            )
        self.assertTrue(
            torch.equal(actual_gate, bmm(inputs, gate[ids].reshape(8, 11, 3)))
        )
        self.assertTrue(torch.equal(actual_up, bmm(inputs, up[ids].reshape(8, 11, 3))))
        self.assertEqual([x.shape[-1] for x, _ in calls], [4, 4, 4, 4, 3, 3])
        for block, start in enumerate((0, 4, 8)):
            stop = min(start + 4, 11)
            for offset, bank in ((0, gate), (1, up)):
                x, y = calls[2 * block + offset]
                self.assertTrue(torch.equal(x, inputs[:, :, start:stop]))
                self.assertTrue(
                    torch.equal(
                        y, bank[:, start:stop, :][ids].reshape(8, stop - start, 3)
                    )
                )
        self.assertEqual([h["work_div"] for h in hints], [{"R": 8}] * 6)

    def test_full_gate_up_gathers_are_only_in_the_ordinary_branch(self):
        region = next(
            n
            for n in ast.parse(SOURCE.read_text()).body
            if getattr(n, "name", "") == "_compiled_moe_loop_region"
        )
        branch = next(
            n
            for n in ast.walk(region)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "gate_up_panel is not None"
        )
        full_reads = [
            n
            for n in ast.walk(region)
            if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id in ("gate_dev", "up_dev")
        ]
        self.assertEqual(len(full_reads), 2)
        ordinary = list(ast.walk(ast.Module(body=branch.orelse, type_ignores=[])))
        self.assertTrue(all(n in ordinary for n in full_reads))

    def test_default_region_uses_blocks_without_a_feature_override(self):
        # Meta tensors exercise shipped dispatch and all BMM shapes without
        # allocating the full bank. Numerical device acceptance is separate.
        def make(*shape):
            return torch.empty(shape, device="meta", dtype=torch.bfloat16)

        ids = torch.empty((1, 8), device="meta", dtype=torch.int64)
        weights = make(1, 8)
        ns = load_functions(
            {"_compiled_moe_loop_region", "_decode_gate_up_blocks"},
            _router_probs=lambda *args: weights,
            _topk=lambda *args: (weights, ids),
        )
        shapes = []
        bmm = torch.bmm

        def record_bmm(x, y):
            shapes.append((tuple(x.shape), tuple(y.shape)))
            return bmm(x, y)

        x = make(1, 2816)
        with recorded_hints(), patch.object(torch, "bmm", record_bmm):
            result = ns["_compiled_moe_loop_region"](
                x,
                x,
                None,
                None,
                None,
                make(128, 64),
                make(128, 2816, 704),
                make(128, 2816, 704),
                make(128, 704, 2816),
                8,
                32,
                64,
                1e-6,
            )
        self.assertEqual(tuple(result.shape), (1, 2816))
        self.assertEqual(len(shapes), 9)
        self.assertEqual([shape[1][-1] for shape in shapes], [704] * 8 + [2816])
        self.assertTrue(all(shape[0][0:2] == (8, 1) for shape in shapes))

    def test_default_panel_and_supported_shape_fallbacks(self):
        choose = load_functions({"_decode_gate_up_panel"})["_decode_gate_up_panel"]
        fp16 = (torch.float16,) * 4
        self.assertEqual(choose(2816, 704, fp16, True), 704)
        self.assertIsNone(choose(2816, 704, fp16, False))
        self.assertIsNone(choose(1408, 704, fp16, True))
        self.assertIsNone(choose(2816, 768, fp16, True))
        self.assertEqual(
            choose(2816, 704, (torch.bfloat16,) * 4, True),
            choose(2816, 704, fp16, True),
        )
        self.assertIsNone(choose(2816, 704, (torch.float32,) * 4, True))
        self.assertIsNone(choose(2816, 704, (*fp16[:3], torch.float32), True))
        # Each 16-bit format is supported, but mixing formats is not.
        self.assertIsNone(choose(2816, 704, (*fp16[:3], torch.bfloat16), True))
        off = load_functions({"_decode_gate_up_panel"}, _DECODE_GATE_UP_K_PANEL=None)
        self.assertIsNone(off["_decode_gate_up_panel"](2816, 704, fp16, True))
        invalid = load_functions({"_decode_gate_up_panel"}, _DECODE_GATE_UP_K_PANEL=123)
        with self.assertRaisesRegex(ValueError, "Unsupported decode block width"):
            invalid["_decode_gate_up_panel"](2816, 704, fp16, True)


if __name__ == "__main__":
    unittest.main()
