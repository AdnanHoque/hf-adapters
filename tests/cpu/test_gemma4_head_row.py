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


def run_prefill_branch(source, namespace):
    """Execute the shipped prefill branch, not a second generation loop."""
    parsed = ast.parse(source.read_text())
    generate = next(n for n in parsed.body if getattr(n, "name", "") == "generate")
    branch = next(
        n
        for n in ast.walk(generate)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "i == 0"
    )
    setup = [
        n
        for n in generate.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in {"prefill_driver", "forward_row_kwargs"}
            for t in n.targets
        )
    ]
    ns = {
        "torch": torch,
        "DEVICE": "cpu",
        "model": object(),
        "input_ids": torch.arange(128).reshape(1, 128),
        "position_ids": torch.arange(128).reshape(1, 128),
        "batch_size": 1,
        "padded_len": 128,
        "prefill_kv_len": 128,
        "query_chunk_size": 64,
        "chunked_prefill": True,
        "prompt_offsets": torch.tensor([0]),
        "model_d_type": torch.float32,
        "key_caches": [object()],
        "value_caches": [object()],
        "_prefill_cache_inputs": lambda caches, *args: caches,
        "build_prefill_mask": lambda *args, **kwargs: torch.zeros(1),
        "make_cache_index": lambda start, length, *args: torch.arange(
            start, start + length
        ),
        "prefill_fn": None,
        "run_forward_fn": None,
        "normalized_token_inputs": {},
        "_generation_last_hidden_row_only": None,
        "_prefill_last_row_only": True,
    }
    ns.update(namespace)
    exec(
        compile(ast.Module(setup + branch.body, type_ignores=[]), str(source), "exec"),
        ns,
    )
    return ns


class GemmaHeadRowTests(unittest.TestCase):
    def test_actual_prefill_driver_gets_request_and_retains_cache_inputs(self):
        for callback in (False, True):
            with self.subTest(callback=callback):
                seen = []

                def forward(
                    model, input_ids, *args, _last_hidden_row_only=False, **kwargs
                ):
                    seen.append(
                        (input_ids.shape[1], _last_hidden_row_only, args, kwargs)
                    )
                    rows = input_ids[:, -1:] if _last_hidden_row_only else input_ids
                    return rows[..., None].expand(-1, -1, 11).float()

                options = load_function(
                    "hf_common.py", "_generation_forward_options", {"inspect": inspect}
                )
                ns = run_prefill_branch(
                    ADAPTERS / "hf_common.py",
                    {
                        "run_forward_fn": None if callback else forward,
                        "prefill_fn": forward if callback else None,
                        "_generation_forward_options": options,
                    },
                )
                self.assertEqual(
                    [(n, flag) for n, flag, _, _ in seen],
                    [(128, True)] if callback else [(64, True), (64, True)],
                )
                for _, _, args, kwargs in seen:
                    self.assertIs(
                        kwargs["key_caches"] if callback else args[2], ns["key_caches"]
                    )
                    self.assertIs(
                        kwargs["value_caches"] if callback else args[3],
                        ns["value_caches"],
                    )
                self.assertTrue(
                    torch.equal(ns["next_logits"], torch.full((1, 11), 127.0))
                )
                self.assertEqual(ns["current_cache_len"], 128)

    def test_unsupported_prefill_callback_ignores_unused_supported_text_driver(self):
        def unused(*args, _last_hidden_row_only=False):
            raise AssertionError("Custom prefill owns this path")

        def callback(**kwargs):
            self.assertNotIn("_last_hidden_row_only", kwargs)
            return torch.zeros(1, 128, 11)

        options = load_function(
            "hf_common.py", "_generation_forward_options", {"inspect": inspect}
        )
        ns = {
            "prefill_fn": callback,
            "run_forward_fn": unused,
            "_generation_forward_options": options,
        }
        run_prefill_branch(ADAPTERS / "hf_common.py", ns)
        with self.assertRaisesRegex(ValueError, "must declare"):
            run_prefill_branch(
                ADAPTERS / "hf_common.py",
                {**ns, "_generation_last_hidden_row_only": True},
            )

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
        self.assertEqual(options(supported), {"_last_hidden_row_only": True})
        self.assertEqual(options(supported, False), {})
        for driver in (None, swallowed, positional, same_name_kwargs):
            self.assertEqual(options(driver), {})
            self.assertEqual(options(driver, False), {})
            with self.assertRaises(ValueError):
                options(driver, True)

    def test_prefill_only_and_default_automatic(self):
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
            None,
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
