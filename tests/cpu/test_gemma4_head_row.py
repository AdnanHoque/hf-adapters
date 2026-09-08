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

"""Execute the shipped row-selection functions without model downloads.

These CPU tests prove row selection and request plumbing, not device numerical
acceptance: changing a matmul's row count may change its reduction schedule.
"""

import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ADAPTERS = Path(__file__).resolve().parents[2] / "hf_adapters"


def load_function(filename, name, namespace):
    source = ADAPTERS / filename
    node = next(
        n for n in ast.parse(source.read_text()).body if getattr(n, "name", "") == name
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


class GemmaHeadRowTests(unittest.TestCase):
    def test_row_identity_backbone_cache_arguments_and_softcap(self):
        for cap in (None, 30.0):
            with self.subTest(cap=cap):
                hidden = (torch.arange(2 * 7 * 5) % 13).reshape(2, 7, 5).double()
                weight = (torch.arange(5 * 11) % 7).reshape(5, 11).double()
                rows, backbone_args = [], []
                key_cache, value_cache, index = object(), object(), object()

                def backbone(*args):
                    backbone_args.append(args)
                    return hidden

                def head(x):
                    rows.append(x.shape[1])
                    return x @ weight

                model = SimpleNamespace(
                    lm_head=head, config=SimpleNamespace(final_logit_softcapping=cap)
                )
                forward = load_function(
                    "hf_gemma4.py",
                    "_run_forward",
                    {
                        "torch": torch,
                        "text_config": lambda c: c,
                        "_run_backbone_forward": backbone,
                    },
                )
                args = (
                    model,
                    torch.ones(2, 7),
                    None,
                    None,
                    key_cache,
                    value_cache,
                    index,
                )
                full = forward(*args)
                one = forward(*args, _last_hidden_row_only=True)
                self.assertEqual(rows, [7, 1])
                self.assertTrue(torch.equal(one[:, 0], full[:, -1]))
                expected = hidden[:, -1:] @ weight
                if cap is not None:
                    expected = torch.tanh(expected / cap) * cap
                self.assertTrue(torch.equal(one, expected))
                self.assertEqual(one.shape, (2, 1, 11))
                for seen in backbone_args:
                    self.assertIs(seen[4], key_cache)
                    self.assertIs(seen[5], value_cache)
                    self.assertIs(seen[6], index)

    def test_request_requires_an_explicit_keyword_parameter(self):
        options = load_function(
            "hf_common.py", "_generation_forward_options", {"inspect": inspect}
        )

        def supported(*args, _last_hidden_row_only=False):
            pass

        def swallowed(*args, **kwargs):
            pass

        def positional(_last_hidden_row_only, /):
            pass

        def same_name_kwargs(**_last_hidden_row_only):
            pass

        self.assertEqual(options(supported, True), {"_last_hidden_row_only": True})
        for driver in (swallowed, positional, same_name_kwargs):
            self.assertEqual(options(driver, False), {})
            with self.assertRaises(ValueError):
                options(driver, True)

    def test_prefill_only_and_default_off(self):
        generate = next(
            n
            for n in ast.parse((ADAPTERS / "hf_common.py").read_text()).body
            if getattr(n, "name", "") == "generate"
        )
        names = [n.arg for n in generate.args.kwonlyargs]
        self.assertIs(
            generate.args.kw_defaults[
                names.index("_generation_last_hidden_row_only")
            ].value,
            False,
        )
        calls = sorted(
            [
                n
                for n in ast.walk(generate)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "run_forward_fn"
            ],
            key=lambda n: n.lineno,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [ast.unparse(k.value) for k in calls[0].keywords if k.arg is None],
            ["forward_row_kwargs"],
        )
        self.assertFalse([k for k in calls[1].keywords if k.arg is None])
        guard = next(
            n
            for n in ast.walk(generate)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_generation_forward_options"
        )
        self.assertLess(guard.lineno, calls[0].lineno)


if __name__ == "__main__":
    unittest.main()
