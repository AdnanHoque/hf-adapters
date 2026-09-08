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

"""CPU checks of output-column blocking; LX placement is a device gate."""

import ast
import unittest
from unittest.mock import patch

import torch
from _gemma4_decode_perf_helpers import SOURCE, load_functions, recorded_hints


class RecordedBank:
    def __init__(self, tensor, events):
        self.tensor, self.events = tensor, events
        self.shape = tensor.shape

    def __getitem__(self, key):
        if isinstance(key, tuple):
            sliced = self.tensor[key]
            self.events.append(
                ("slice", key[-1].start, key[-1].stop, tuple(sliced.shape))
            )
            return RecordedBank(sliced, self.events)
        selected = self.tensor[key]
        self.events.append(("select", tuple(selected.shape)))
        return selected


class DownBlockTests(unittest.TestCase):
    def test_columns_tails_repeated_ids_and_full_reductions(self):
        namespace = load_functions({"_decode_down_output_blocks"})
        bank = (torch.arange(8 * 5 * 11).reshape(8, 5, 11) % 17).double()
        ids = torch.tensor([[0, 0, 7, 7, 3, 6, 1, 2]])
        activated = torch.arange(8 * 5).reshape(8, 1, 5).double() / 8
        expected = torch.bmm(activated, bank[ids].reshape(8, 5, 11))
        events, shapes = [], []
        bmm = torch.bmm

        def record_bmm(x, y):
            shapes.append((tuple(x.shape), tuple(y.shape)))
            return bmm(x, y)

        with recorded_hints() as hints, patch.object(torch, "bmm", record_bmm):
            result = namespace["_decode_down_output_blocks"](
                activated, RecordedBank(bank, events), ids, 4
            )
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(
            [e[1:3] for e in events if e[0] == "slice"], [(0, 4), (4, 8), (8, 11)]
        )
        self.assertEqual(
            [e[1] for e in events if e[0] == "select"],
            [(1, 8, 5, 4), (1, 8, 5, 4), (1, 8, 5, 3)],
        )
        self.assertEqual([s[1][1] for s in shapes], [5, 5, 5])
        self.assertEqual([h["work_div"] for h in hints], [{"R": 8, "H": 1}] * 3)

    def test_full_gather_is_guarded_out_when_output_blocks_are_requested(self):
        region = next(
            n
            for n in ast.parse(SOURCE.read_text()).body
            if getattr(n, "name", "") == "_compiled_moe_loop_region"
        )
        guards = [
            n
            for n in ast.walk(region)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "down_panel is None"
        ]
        self.assertGreaterEqual(len(guards), 1)
        full_reads = [
            n
            for n in ast.walk(region)
            if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id == "down_dev"
        ]
        self.assertTrue(full_reads)
        # A later gate/up-blocking PR can add another ordinary-down branch.
        # Every such read must remain guarded, not just the first one found.
        guarded_nodes = [
            node
            for guard in guards
            for statement in guard.body
            for node in ast.walk(statement)
        ]
        self.assertTrue(all(read in guarded_nodes for read in full_reads))

    def test_default_region_uses_blocks_without_a_feature_override(self):
        # Meta tensors exercise shipped dispatch and all BMM shapes without
        # allocating the full bank. Numerical device acceptance is separate.
        def make(*shape):
            return torch.empty(shape, device="meta", dtype=torch.float16)

        ids = torch.empty((1, 8), device="meta", dtype=torch.int64)
        weights = make(1, 8)
        ns = load_functions(
            {"_compiled_moe_loop_region", "_decode_down_output_blocks"},
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
        self.assertEqual(len(shapes), 5)
        self.assertEqual(
            [shape[1][-1] for shape in shapes], [704, 704, 1024, 1024, 768]
        )
        self.assertTrue(all(shape[0][0:2] == (8, 1) for shape in shapes))

    def test_default_panel_and_supported_shape_fallbacks(self):
        choose = load_functions({"_decode_down_panel"})["_decode_down_panel"]
        fp16 = (torch.float16,) * 4
        self.assertEqual(choose(2816, 704, fp16, True), 1024)
        self.assertIsNone(choose(2816, 704, fp16, False))
        self.assertIsNone(choose(1408, 704, fp16, True))
        self.assertIsNone(choose(2816, 768, fp16, True))
        self.assertIsNone(choose(2816, 704, (torch.bfloat16,) * 4, True))
        self.assertIsNone(choose(2816, 704, (*fp16[:3], torch.float32), True))
        off = load_functions({"_decode_down_panel"}, _DECODE_DOWN_OUTPUT_PANEL=None)
        self.assertIsNone(off["_decode_down_panel"](2816, 704, fp16, True))
        invalid = load_functions({"_decode_down_panel"}, _DECODE_DOWN_OUTPUT_PANEL=123)
        with self.assertRaisesRegex(ValueError, "Unsupported decode block width"):
            invalid["_decode_down_panel"](2816, 704, fp16, True)


if __name__ == "__main__":
    unittest.main()
