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
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "_DECODE_DOWN_OUTPUT_PANEL is None"
        ]
        self.assertEqual(len(guards), 1)
        full_reads = [
            n
            for n in ast.walk(region)
            if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id == "down_dev"
        ]
        self.assertEqual(len(full_reads), 1)
        self.assertIn(full_reads[0], list(ast.walk(guards[0])))
        self.assertNotIn("_DECODE_GATE_UP_K_PANEL", ast.unparse(region))

    def test_explicit_request_without_route_schedule_declines(self):
        namespace = load_functions(
            {"_compiled_moe_loop_region"}, _DECODE_DOWN_OUTPUT_PANEL=1024
        )
        with (
            recorded_hints(),
            self.assertRaisesRegex(ValueError, "Down blocks require"),
        ):
            namespace["_compiled_moe_loop_region"](
                None,
                torch.zeros(1, 4),
                None,
                None,
                None,
                None,
                torch.empty(8, 4, 3),
                None,
                None,
                8,
                32,
                2,
                1e-6,
            )


if __name__ == "__main__":
    unittest.main()
