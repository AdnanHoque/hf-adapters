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

"""Focused tests of the route schedule and its scoped compiler requirement."""

import contextlib
import unittest
from types import SimpleNamespace

import torch
from _gemma4_decode_perf_helpers import load_functions, recorded_hints


class GemmaDecodeScheduleTests(unittest.TestCase):
    def run_region(self, enabled, tokens=1, routes=8):
        ids = torch.tensor([[0, 0, 1, 3, 7, 7, 4, 6]])[:, :routes].expand(tokens, -1)
        weights = torch.ones(tokens, routes, dtype=torch.float64)
        namespace = load_functions(
            {"_compiled_moe_loop_region"},
            _DECODE_ROUTE_SCHEDULE=enabled,
            _router_probs=lambda *args: weights,
            _topk=lambda *args: (weights, ids),
        )
        x = torch.arange(tokens * 4, dtype=torch.float64).reshape(tokens, 4) / 8
        gate = (torch.arange(8 * 4 * 3).reshape(8, 4, 3) % 7).double() / 16
        up = gate + 0.25
        down = gate.transpose(1, 2).contiguous()
        with recorded_hints() as hints:
            result = namespace["_compiled_moe_loop_region"](
                x,
                x,
                None,
                None,
                None,
                torch.ones(8, 1),
                gate,
                up,
                down,
                routes,
                32,
                2,
                1e-6,
            )
        return result, hints

    def test_default_and_r8_have_equal_cpu_arithmetic_with_repeated_ids(self):
        ordinary, default_hints = self.run_region(False)
        routed, routed_hints = self.run_region(True)
        self.assertTrue(torch.equal(ordinary, routed))
        self.assertFalse([h for h in default_hints if "work_div" in h])
        self.assertEqual(
            [h["work_div"] for h in routed_hints if "work_div" in h], [{"R": 8}]
        )

    def test_unsupported_token_and_route_counts_decline(self):
        for tokens, routes in ((2, 8), (1, 3)):
            with self.subTest(tokens=tokens, routes=routes):
                with self.assertRaisesRegex(ValueError, "one token and eight routes"):
                    self.run_region(True, tokens, routes)
                result, _ = self.run_region(False, tokens, routes)
                self.assertEqual(result.shape, (tokens, 4))

    def test_compiler_option_is_scoped_and_restored_on_error(self):
        active, observed = {}, []

        @contextlib.contextmanager
        def patch_options(options):
            old = active.copy()
            active.update(options)
            try:
                yield
            finally:
                active.clear()
                active.update(old)

        def decode(*args):
            observed.append(active.copy())
            raise RuntimeError("test failure")

        namespace = load_functions(
            {"forward"},
            _DECODE_ROUTE_SCHEDULE=True,
            optional_spyre_config_patch=patch_options,
        )
        block = SimpleNamespace(_compiled_decode=decode)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            namespace["forward"](
                block, torch.zeros(1, 1, 4), None, None, None, None, None, None
            )
        self.assertEqual(observed, [{"indexed_selection_consumer_layout": True}])
        self.assertEqual(active, {})

    def test_default_decode_does_not_touch_compiler_configuration(self):
        namespace = load_functions(
            {"forward"},
            optional_spyre_config_patch=lambda options: self.fail(
                "default patched config"
            ),
        )
        expected = (object(), object(), object())
        block = SimpleNamespace(_compiled_decode=lambda *args: expected)
        self.assertEqual(
            namespace["forward"](
                block, torch.zeros(1, 1, 4), None, None, None, None, None, None
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
