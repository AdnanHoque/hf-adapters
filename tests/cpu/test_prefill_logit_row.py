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

"""Movement/order tests of the shipped helper; no device performance claim.

Load the actual function body without importing optional model packages. The
Spyre compiled-copy regression and HF generation checks remain device tests.
"""

import ast
import unittest
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[2] / "hf_adapters/hf_common.py"


def load_helper():
    module = ast.parse(SOURCE.read_text())
    function = next(
        n for n in module.body if getattr(n, "name", "") == "_prefill_next_logits"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace[function.name]


class TrackedTensor:
    def __init__(self, value, events):
        self.value = value
        self.events = events

    def __getitem__(self, key):
        return TrackedTensor(self.value[key], self.events)

    def clone(self):
        self.events.append(("clone", tuple(self.value.shape)))
        return TrackedTensor(self.value.clone(), self.events)

    def to(self, device):
        self.events.append((device, tuple(self.value.shape)))
        return TrackedTensor(self.value.to(device), self.events)


class PrefillLogitRowTests(unittest.TestCase):
    def test_last_row_exact_for_multiple_batches_and_dtypes(self):
        helper = load_helper()
        for dtype in (torch.float32, torch.bfloat16):
            for rows in (1, 7, 64):
                with self.subTest(dtype=dtype, rows=rows):
                    x = (
                        (torch.arange(2 * rows * 11) % 64)
                        .reshape(2, rows, 11)
                        .to(dtype)
                    )
                    self.assertTrue(
                        torch.equal(helper(x, last_row_only=False), helper(x))
                    )
                    self.assertEqual(helper(x, last_row_only=True).shape, (2, 11))

    def test_materializes_one_row_before_cpu_transfer(self):
        events = []
        x = TrackedTensor(torch.arange(2 * 9 * 13).reshape(2, 9, 13), events)
        out = load_helper()(x)
        self.assertEqual(events, [("clone", (2, 1, 13)), ("cpu", (2, 1, 13))])
        self.assertTrue(torch.equal(out.value, x.value[:, -1, :]))

    def test_off_retains_full_transfer_without_clone(self):
        events = []
        x = TrackedTensor(torch.zeros(2, 9, 13), events)
        load_helper()(x, last_row_only=False)
        self.assertEqual(events, [("cpu", (2, 9, 13))])

    def test_noncontiguous_input_and_distinct_storage(self):
        x = torch.arange(3 * 7 * 5).reshape(3, 5, 7).transpose(1, 2)
        out = load_helper()(x, last_row_only=True)
        self.assertTrue(torch.equal(out, x[:, -1, :]))
        self.assertNotEqual(
            out.untyped_storage().data_ptr(), x.untyped_storage().data_ptr()
        )
        self.assertEqual(
            out.untyped_storage().nbytes(), out.numel() * out.element_size()
        )

    def test_only_prefill_uses_helper_and_flag_defaults_on(self):
        generate = next(
            n
            for n in ast.parse(SOURCE.read_text()).body
            if getattr(n, "name", "") == "generate"
        )
        names = [n.arg for n in generate.args.kwonlyargs]
        self.assertIs(
            generate.args.kw_defaults[names.index("_prefill_last_row_only")].value,
            True,
        )
        calls = [
            n
            for n in ast.walk(generate)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_prefill_next_logits"
        ]
        self.assertEqual(len(calls), 1)
        prefill = next(
            n
            for n in ast.walk(generate)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "i == 0"
        )
        self.assertIn(
            calls[0], list(ast.walk(ast.Module(body=prefill.body, type_ignores=[])))
        )
        self.assertNotIn(
            calls[0], list(ast.walk(ast.Module(body=prefill.orelse, type_ignores=[])))
        )


if __name__ == "__main__":
    unittest.main()
