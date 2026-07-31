from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from torch_dae.models.panns import PUBLIC_MODEL_IDENTITIES, load_audioset_labels


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_three_fixed_public_identities_are_dependency_free() -> None:
    assert [identity.model_id for identity in PUBLIC_MODEL_IDENTITIES] == [
        "panns-cnn14-16k-map-0438",
        "panns-resnet38-map-0434",
        "panns-wavegram-logmel-cnn14-map-0439",
    ]
    assert [identity.upstream_class for identity in PUBLIC_MODEL_IDENTITIES] == [
        "Cnn14_16k",
        "ResNet38",
        "Wavegram_Logmel_Cnn14",
    ]
    assert [identity.sample_rate for identity in PUBLIC_MODEL_IDENTITIES] == [
        16_000,
        32_000,
        32_000,
    ]
    assert "torch" not in sys.modules
    assert "torchlibrosa" not in sys.modules


def test_root_import_and_missing_dependency_diagnostic(repo_root: Path) -> None:
    root_import = subprocess.run(
        [sys.executable, "-c", "import torch_dae; import torch_dae.models.panns"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert root_import.returncode == 0, root_import.stderr

    missing_runtime = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; "
                "from torch_dae.models.panns import PannsCnn14_16kMap0438"
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_runtime.returncode != 0
    assert "isolated model environment" in missing_runtime.stderr
    assert "torch==2.13.0" in missing_runtime.stderr


def test_audioset_metadata_is_authoritative_and_ordered(repo_root: Path) -> None:
    labels = load_audioset_labels()
    assert len(labels) == 527
    assert [label.index for label in labels] == list(range(527))
    assert len({label.mid for label in labels}) == 527
    assert labels[0].mid == "/m/09x0r"
    assert labels[0].display_name == "Speech"
    assert (
        sha256(repo_root / "src/torch_dae/models/panns/resources/class_labels_indices.csv")
        == "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429"
    )


def test_vendor_surface_contains_only_required_upstream_symbols(repo_root: Path) -> None:
    path = repo_root / "src/torch_dae/models/panns/_vendor/models.py"
    tree = ast.parse(path.read_text())
    definitions = {
        node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert definitions == {
        "init_layer",
        "init_bn",
        "ConvBlock",
        "_resnet_conv3x3",
        "_resnet_conv1x1",
        "_ResnetBasicBlock",
        "_ResNet",
        "ResNet38",
        "ConvPreWavBlock",
        "Wavegram_Logmel_Cnn14",
        "Cnn14_16k",
    }
    utilities = ast.parse(
        (repo_root / "src/torch_dae/models/panns/_vendor/pytorch_utils.py").read_text()
    )
    assert [node.name for node in utilities.body if isinstance(node, ast.FunctionDef)] == [
        "do_mixup"
    ]
    text = path.read_text()
    assert text.count("logits = self.fc_audioset(x)") == 3
    assert text.count("clipwise_output = torch.sigmoid(logits)") == 3
    assert "torch.logit" not in text


def test_source_provenance_matches_integrated_files(repo_root: Path) -> None:
    provenance_path = (
        repo_root / "onboarding_reports/panns-audioset-three-tuple/integrate/source-provenance.json"
    )
    if not provenance_path.is_file():
        return
    provenance = json.loads(provenance_path.read_text())
    for record in provenance["files"]:
        integrated = repo_root / record["integrated_path"]
        assert integrated.is_file()
        assert integrated.stat().st_size == record["integrated_size_bytes"]
        assert sha256(integrated) == record["integrated_sha256"]
