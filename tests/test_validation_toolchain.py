from __future__ import annotations

import subprocess
import sys
import zipfile


def test_twine_accepts_core_metadata_25(tmp_path):
    wheel = tmp_path / "synthetic_metadata-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "synthetic_metadata-0.0.0.dist-info/METADATA",
            (
                "Metadata-Version: 2.5\nName: synthetic-metadata\nVersion: 0.0.0\n"
                "Summary: Scientifically meaningless validation fixture\n"
                "Description-Content-Type: text/plain\n\nSynthetic validator regression.\n"
            ),
        )
        archive.writestr(
            "synthetic_metadata-0.0.0.dist-info/WHEEL",
            (
                "Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\n"
                "Tag: py3-none-any\n"
            ),
        )
    result = subprocess.run(
        [sys.executable, "-m", "twine", "check", str(wheel)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_validation_commands_pin_interpreter_per_invocation(repo_root):
    workflow = (repo_root / ".github/workflows/ci.yml").read_text()
    matrix = workflow.split("  tests:")[1].split("  coverage:")[0]
    assert (
        "uv run --python ${{ matrix.python-version }} --all-groups --extra profiling --frozen "
        "pytest -q" in matrix
    )
    assert "uv run python" not in matrix

    # `docs/development/testing.md` documents the root control-plane sequence, which now also
    # exercises the Profiling v1 `profiling` optional-dependency group (numpy/psutil/codecarbon);
    # `skills/audio-model-onboarding/references/workflow-overview.md` documents the onboarding
    # skill's own sequence, which never touches profiling and keeps the plain invocation.
    testing_doc = (repo_root / "docs/development/testing.md").read_text()
    assert "uv run --python 3.11 --all-groups --extra profiling --frozen" in testing_doc

    onboarding_doc = (
        repo_root / "skills/audio-model-onboarding/references/workflow-overview.md"
    ).read_text()
    assert "uv run --python 3.11 --all-groups --frozen" in onboarding_doc


def test_local_wheel_builder_uses_active_control_plane_python(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from torch_dae.environment.manager import EnvironmentManager

    captured = []
    manager = EnvironmentManager(tmp_path)
    manager.executor = SimpleNamespace(run=lambda command, **kwargs: captured.append(command))
    monkeypatch.setattr("torch_dae.environment.manager.valid_single_wheel", lambda path: path)
    manager._build_local_wheel_with_backend(tmp_path)
    command = captured[0]
    assert command[command.index("--python") + 1] == sys.executable
    assert "--no-build-isolation" in command
    assert "--offline" in command
