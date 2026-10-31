"""The published Output schema must be one URL, never an array.

Runs on CPU with no weights: Cog's schema generation only reads the annotated
source of the predictor, so a machine without the model assets can still prove
what shape Replicate is told to expect. `import run` is also checked to stay
torch-free, otherwise this test could not run in the ordinary test suite.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import warnings

import pytest

from cog._inspector import create_predictor

ROOT = Path(__file__).resolve().parents[1]


def output_schema(module_name: str, class_name: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return create_predictor(module_name, class_name).output.json_type()


@pytest.fixture
def load_module(monkeypatch):
    def load(directory: Path, module_name: str):
        monkeypatch.syspath_prepend(str(directory))
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        return importlib.import_module(module_name)
    return load


def test_runner_output_is_one_uri_not_an_array():
    schema = output_schema("run", "Runner")
    assert schema == {"title": "Output", "type": "string", "format": "uri"}
    assert schema["type"] != "array"
    assert "items" not in schema


def test_importing_the_predictor_does_not_load_torch():
    # Keeps schema generation and this suite runnable without the CUDA stack.
    loaded = subprocess.run(
        [sys.executable, "-c", "import sys, run; print('torch' in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert loaded.stdout.strip() == "False"


def test_runner_inputs_keep_the_five_field_contract():
    info = create_predictor("run", "Runner")
    assert list(info.inputs) == ["image", "prompt", "output_zip", "confidence", "image_size"]


def test_a_list_output_would_be_reported_as_an_array(tmp_path, load_module):
    # Proof the assertion above has teeth: List[Path] yields an array schema,
    # which is the shape that produced ["https://..."] from Replicate.
    (tmp_path / "list_runner.py").write_text(
        "from cog import BaseRunner, Path\n"
        "class Runner(BaseRunner):\n"
        "    def run(self, image: Path) -> list[Path]:\n"
        "        return [image]\n",
        encoding="utf-8",
    )
    load_module(tmp_path, "list_runner")
    assert output_schema("list_runner", "Runner")["type"] == "array"


def test_legacy_predict_and_run_styles_generate_the_same_output_schema(tmp_path, load_module):
    # The `predict: "predict.py:Predictor"` + BasePredictor API is not a fix for
    # the array output: both interfaces declare the identical scalar schema.
    # The array comes from coglet's progress webhook, not from this schema.
    (tmp_path / "predict.py").write_text(
        "from cog import BasePredictor, Path\n"
        "class Predictor(BasePredictor):\n"
        "    def predict(self, image: Path) -> Path:\n"
        "        return image\n",
        encoding="utf-8",
    )
    (tmp_path / "run.py").write_text(
        "from cog import BaseRunner, Path\n"
        "class Runner(BaseRunner):\n"
        "    def run(self, image: Path) -> Path:\n"
        "        return image\n",
        encoding="utf-8",
    )
    load_module(tmp_path, "predict")
    load_module(tmp_path, "run")
    assert output_schema("predict", "Predictor") == output_schema("run", "Runner")
