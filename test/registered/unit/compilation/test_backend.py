import os
import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_BUILTIN_BACKEND_MODULES = (
    "sglang.srt.compilation.cuda_piecewise_backend",
    "sglang.srt.compilation.npu_piecewise_backend",
    "sglang.srt.compilation.xpu_piecewise_backend",
)


def _run_isolated(code: str, *, use_cpu: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env.pop("SGLANG_PLATFORM", None)
    env.pop("SGLANG_USE_CPU_ENGINE", None)
    if use_cpu:
        env["SGLANG_USE_CPU_ENGINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo_root / "python"), env.get("PYTHONPATH")))
    )

    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_cpu_import_skips_builtin_piecewise_backends():
    modules = repr(_BUILTIN_BACKEND_MODULES)
    code = f"""
import sys

import sglang.srt.compilation.backend

unexpected = [name for name in {modules} if name in sys.modules]
assert not unexpected, unexpected
"""

    _run_isolated(code, use_cpu=True)


def test_out_of_tree_piecewise_backend_takes_precedence():
    modules = repr(_BUILTIN_BACKEND_MODULES)
    code = f"""
import sys

import sglang.srt.platforms as platforms


class OOTBackend:
    def __init__(self, *args):
        self.args = args


class OOTPlatform:
    def is_out_of_tree(self):
        return True

    def get_piecewise_backend_cls(self):
        return OOTBackend


platforms._current_platform = OOTPlatform()

from sglang.srt.compilation import backend

unexpected = [name for name in {modules} if name in sys.modules]
assert not unexpected, unexpected


def unexpected_in_tree_probe():
    raise AssertionError("in-tree platform probe must not run for an OOT platform")


backend.is_xpu = unexpected_in_tree_probe
backend.is_npu = unexpected_in_tree_probe
args = tuple(object() for _ in range(9))
selected = backend.make_backend(*args)

assert isinstance(selected, OOTBackend)
assert selected.args == args
unexpected = [name for name in {modules} if name in sys.modules]
assert not unexpected, unexpected
"""

    _run_isolated(code)
