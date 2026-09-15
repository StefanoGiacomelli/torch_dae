from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from torch_dae.core.checkpoint import ResolvedCheckpoint
from torch_dae.core.errors import UnsupportedCapabilityError
from torch_dae.environment.results import EnvironmentVerificationResult
from torch_dae.environment.verification import TensorObservation, VerificationReport
from torch_dae.runtime_executor import execute_runtime_verification, validate_runtime_report
from torch_dae.runtime_verification import RuntimeVerificationTarget
from torch_dae.runtime_worker import execute_request, run_checks


@pytest.fixture
def target(valid_fixture_dir):
    data = json.loads(
        (valid_fixture_dir / "runtime-verification-target.synthetic.json").read_text()
    )
    data.update(
        required_check_ids=["checkpoint_strict_load", "output_contract"],
        optional_check_ids=["optional_capability"],
    )
    return RuntimeVerificationTarget.model_validate(data)


class SyntheticProvider:
    def __init__(self, *args):
        self.tensor_observations: list[TensorObservation] = []
        self.embedding_results: list[str] = []
        self.calls = []

    def check(self, name):
        self.calls.append(name)
        if name == "optional_capability":
            raise UnsupportedCapabilityError("Synthetic optional operation unavailable.")
        return "Synthetic strict load succeeded."


def request(target, valid_fixture_dir):
    return {
        "target": target.model_dump(mode="json"),
        "environment": {"environment_fingerprint": "a" * 64, "platform": "synthetic"},
        "checkpoint": {"path": "/synthetic/checkpoint", "sha256": "b" * 64},
        "checkpoint_materialization": {"path": "synthetic.json", "sha256": "c" * 64},
        "controls": {},
    }


def worker_report(monkeypatch, target, valid_fixture_dir, provider=SyntheticProvider):
    import torch_dae.runtime_worker as worker

    with monkeypatch.context() as patch:
        patch.setattr(
            worker.importlib,
            "import_module",
            lambda name: SimpleNamespace(SyntheticModel=object, Provider=provider),
        )
        return execute_request(request(target, valid_fixture_dir))


def test_dispatches_each_required_and_optional_check_once(target):
    provider = SyntheticProvider()
    checks = run_checks(target, provider, {})
    assert provider.calls == [*target.required_check_ids, *target.optional_check_ids]
    assert [c.status for c in checks] == ["passed", "passed", "unsupported"]


def test_worker_strict_success_optional_unsupported(monkeypatch, target, valid_fixture_dir):
    report = worker_report(monkeypatch, target, valid_fixture_dir)
    assert report.successful
    assert report.checks[0].name == "checkpoint_strict_load"
    assert report.checks[0].status == "passed"
    assert report.unsupported_capabilities == ("optional_capability",)
    assert "Synthetic optional operation unavailable." in report.known_limitations


def test_worker_strict_failure_is_failed_required_check(monkeypatch, target, valid_fixture_dir):
    class FailingProvider(SyntheticProvider):
        def check(self, name):
            if name == "checkpoint_strict_load":
                raise RuntimeError("Missing key(s) in state_dict")
            return super().check(name)

    report = worker_report(monkeypatch, target, valid_fixture_dir, FailingProvider)
    assert not report.successful
    assert report.checks[0].status == "failed"
    assert "Missing key" in report.checks[0].details


@pytest.mark.parametrize(
    "mutation", ["omit", "undeclared", "failed_passed", "unsupported_required"]
)
def test_report_cannot_claim_incomplete_success(monkeypatch, target, valid_fixture_dir, mutation):
    data = worker_report(monkeypatch, target, valid_fixture_dir).model_dump(mode="json")
    if mutation == "omit":
        data["checks"].pop(0)
    elif mutation == "undeclared":
        data["checks"].append({"name": "undeclared", "status": "passed"})
    else:
        data["checks"][0]["status"] = "failed" if mutation == "failed_passed" else "unsupported"
    with pytest.raises(ValidationError):
        VerificationReport.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "environment_fingerprint",
        "checkpoint_sha256",
        "environment_id",
        "checkpoint_id",
        "required_check_ids",
    ],
)
def test_report_identity_mismatch_rejected(monkeypatch, target, valid_fixture_dir, field):
    report = worker_report(monkeypatch, target, valid_fixture_dir)
    value = ("wrong",) if field == "required_check_ids" else "d" * 64
    report = report.model_copy(update={field: value})
    with pytest.raises(ValueError):
        validate_runtime_report(
            report, target, environment_fingerprint="a" * 64, checkpoint_sha256="b" * 64
        )


@pytest.mark.parametrize("mismatch", [None, "environment", "checkpoint"])
def test_executor_selects_managed_interpreter_without_card(
    monkeypatch,
    target,
    valid_fixture_dir,
    tmp_path,
    mismatch,
):
    import torch_dae.runtime_executor as executor

    environment = EnvironmentVerificationResult.model_validate_json(
        (valid_fixture_dir / "environment-verification-result.synthetic.json").read_text()
    )
    environment = environment.model_copy(
        update={
            "environment_id": target.environment_id,
            "environment_spec_sha256": target.environment_spec_sha256,
            "environment_fingerprint": "a" * 64,
        }
    )
    checkpoint_path = tmp_path / "cache/checkpoint.bin"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"synthetic")
    (checkpoint_path.parent / "checkpoint-materialization.json").write_text("{}")
    checkpoint = ResolvedCheckpoint(
        checkpoint_id=target.checkpoint.checkpoint_id,
        sha256="b" * 64,
        path=checkpoint_path,
        immutable=True,
    )
    selected_python = tmp_path / ".torch-dae/environments/selected/bin/python"
    calls = []
    report = worker_report(monkeypatch, target, valid_fixture_dir)
    # Supply observations matching the explicit synthetic target.
    observations = []
    for output in target.expected_outputs:
        observations.append(
            TensorObservation(
                name=output.name,
                role="output",
                component_path="synthetic",
                rank=output.rank,
                shape=[
                    {"name": d, "size": int(d) if d.isdecimal() else 1} for d in output.dimensions
                ],
                dtype="float32",
                device="cpu",
            )
        )
    report = report.model_copy(
        update={
            "tensor_observations": tuple(observations),
            "embedding_results": (target.default_embedding.embedding_id,),
        }
    )

    class Commands:
        def with_report_sink(self, sink):
            return self

        def run(self, command, **kwargs):
            calls.append((command, kwargs))
            incoming = json.loads(Path(command[-2]).read_text())
            updated = report.model_copy(
                update={
                    "checkpoint_materialization": executor.ArtifactEvidence.model_validate(
                        incoming["checkpoint_materialization"]
                    ),
                }
            )
            Path(command[-1]).write_text(updated.model_dump_json())
            return SimpleNamespace(returncode=0, stderr="")

    manager = SimpleNamespace(
        materialize_environment=lambda *a, **k: SimpleNamespace(
            environment_fingerprint="a" * 64, model_dump_json=lambda **k: "{}"
        ),
        verify_environment=lambda *a, **k: environment,
        resolved_environment=lambda *a: SimpleNamespace(
            fingerprint="a" * 64, python_executable=selected_python
        ),
        executor=Commands(),
    )
    monkeypatch.setattr(executor, "validate_runtime_verification_target", lambda t, r: t)
    monkeypatch.setattr(executor, "EnvironmentManager", lambda *a, **k: manager)
    monkeypatch.setattr(
        executor,
        "CheckpointManager",
        lambda *a, **k: SimpleNamespace(ensure_checkpoint=lambda *a, **k: checkpoint),
    )
    if mismatch == "environment":
        environment = environment.model_copy(update={"environment_fingerprint": "d" * 64})
    if mismatch == "checkpoint":
        checkpoint = checkpoint.model_copy(update={"checkpoint_id": "wrong-checkpoint"})
    if mismatch:
        with pytest.raises(ValueError, match="identity mismatch"):
            execute_runtime_verification(target, tmp_path, offline=True)
        assert calls == []
        return
    result = execute_runtime_verification(target, tmp_path, offline=True)
    assert result.report.successful
    assert not (tmp_path / "model_cards").exists()
    command, options = calls[0]
    assert command[:4] == [str(selected_python), "-I", "-m", "torch_dae.runtime_worker"]
    assert options["timeout"] == target.verification_limits.timeout_seconds
    assert options["env_remove"] == ("PYTHONPATH", "PYTHONHOME")
    assert options["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    # Future card evidence consumes the existing strict report without a new format.
    assert VerificationReport.model_validate_json(result.report_path.read_text()).successful


def test_worker_runs_in_a_distinct_interpreter(tmp_path, repo_root, target, valid_fixture_dir):
    """A real child process imports the provider from its selected environment only."""
    import subprocess
    import sys
    import sysconfig
    import venv

    selected = tmp_path / "selected"
    venv.EnvBuilder(with_pip=False).create(selected)
    python = selected / "bin/python"
    site = (
        selected
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    # Reuse control-plane-only dependencies, while keeping a separate interpreter prefix.
    (site / "fixture.pth").write_text(str(sysconfig.get_path("purelib")) + "\n")
    package = site / "torch_dae"
    package.mkdir()
    (package / "__init__.py").write_text(f"__path__.append({str(repo_root / 'src/torch_dae')!r})\n")
    synthetic = package / "synthetic_worker"
    synthetic.mkdir()
    (synthetic / "__init__.py").write_text("class SyntheticModel: pass\n")
    (synthetic / "verification.py").write_text(
        """
import sys
from torch_dae.core.errors import UnsupportedCapabilityError
class Provider:
    tensor_observations = []
    embedding_results = []
    def __init__(self, model_class, target, checkpoint):
        assert sys.prefix == EXPECTED_PREFIX
        assert model_class.__name__ == "SyntheticModel"
        assert "torch" not in sys.modules
    def check(self, name):
        if name == "optional_capability":
            raise UnsupportedCapabilityError("Synthetic unsupported optional check")
        return "Executed in " + sys.prefix
""".replace("EXPECTED_PREFIX", repr(str(selected)))
    )
    data = request(target, valid_fixture_dir)
    data["target"]["public_model_entry_point"] = "torch_dae.synthetic_worker:SyntheticModel"
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(data))
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [str(python), "-I", "-m", "torch_dae.runtime_worker", str(request_path), str(report_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = VerificationReport.model_validate_json(report_path.read_text())
    assert report.successful
    assert report.checks[0].details == "Executed in " + str(selected)
    assert str(selected) != sys.prefix


def test_unknown_required_check_fails_instead_of_disappearing(target):
    data = target.model_dump(mode="json")
    data["required_check_ids"] = ["optional_capability"]
    data["optional_check_ids"] = []
    required = RuntimeVerificationTarget.model_validate(data)
    checks = run_checks(required, SyntheticProvider(), {})
    assert len(checks) == 1 and checks[0].status == "failed"


def test_report_rejects_missing_output_and_embedding(monkeypatch, target, valid_fixture_dir):
    report = worker_report(monkeypatch, target, valid_fixture_dir)
    with pytest.raises(ValueError, match="missing or invalid output"):
        validate_runtime_report(
            report, target, environment_fingerprint="a" * 64, checkpoint_sha256="b" * 64
        )


def test_cli_returns_failed_runtime_report(monkeypatch, target, valid_fixture_dir, tmp_path):
    from typer.testing import CliRunner

    from torch_dae.cli.main import app

    class FailedProvider(SyntheticProvider):
        def check(self, name):
            raise RuntimeError("Synthetic strict load failure")

    report = worker_report(monkeypatch, target, valid_fixture_dir, FailedProvider)
    result = SimpleNamespace(
        report=report,
        report_path=tmp_path / "report.json",
        environment_result=SimpleNamespace(model_dump=lambda **k: {}),
        evidence_directory=tmp_path,
    )
    monkeypatch.setattr(
        "torch_dae.runtime_executor.execute_runtime_verification", lambda *a, **k: result
    )
    path = tmp_path / "target.json"
    path.write_text(target.model_dump_json())
    response = CliRunner().invoke(app, ["model", "verify", "--target", str(path), "--json"])
    assert response.exit_code == 1, response.output
    assert json.loads(response.output)["report"]["verification_status"] == "failed"


@pytest.mark.parametrize("wrong", ["workflow", "phase"])
def test_target_rejects_wrong_handoff_identity(monkeypatch, target, tmp_path, wrong):
    import hashlib

    import torch_dae.onboarding.handoff as handoff_module
    from torch_dae.onboarding.contracts import HandoffStatus, OnboardingPhase
    from torch_dae.runtime_verification import validate_runtime_verification_target

    path = tmp_path / target.accepted_integration_handoff.path
    path.parent.mkdir(parents=True)
    path.write_text("synthetic handoff bytes")
    target = target.model_copy(
        update={
            "accepted_integration_handoff": target.accepted_integration_handoff.model_copy(
                update={
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ),
        }
    )
    monkeypatch.setattr(handoff_module, "validate_workflow", lambda *a, **k: {})
    monkeypatch.setattr(
        handoff_module,
        "load_handoff",
        lambda p: SimpleNamespace(
            workflow_id="wrong-workflow" if wrong == "workflow" else target.workflow_id,
            phase=OnboardingPhase.ANALYZE if wrong == "phase" else OnboardingPhase.INTEGRATE,
            handoff_status=HandoffStatus.ACCEPTED,
        ),
    )
    with pytest.raises(ValueError, match="handoff identity mismatch"):
        validate_runtime_verification_target(target, tmp_path)


def test_generic_control_plane_history_keeps_model_artifacts_strict(tmp_path):
    from torch_dae.onboarding.handoff import (
        SHARED_CONTROL_PLANE_ARTIFACTS,
        _validate_latest_external_artifacts,
    )

    generic = "src/torch_dae/environment/manager.py"
    model = "src/torch_dae/models/panns/model.py"
    environment = "environments/panns-cnn14-16k-map-0438/environment.json"
    assert generic in SHARED_CONTROL_PLANE_ARTIFACTS
    assert model not in SHARED_CONTROL_PLANE_ARTIFACTS
    assert environment not in SHARED_CONTROL_PLANE_ARTIFACTS
    for relative in (generic, model, environment):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed bytes")
    from torch_dae.onboarding.contracts import OnboardingPhase

    errors = _validate_latest_external_artifacts(
        tmp_path,
        {
            relative: (OnboardingPhase.INTEGRATE, "a" * 64)
            for relative in (generic, model, environment)
        },
    )
    assert len(errors) == 2
    assert any(model in error for error in errors)
    assert any(environment in error for error in errors)
