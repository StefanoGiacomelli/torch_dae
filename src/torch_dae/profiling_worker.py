"""Managed-interpreter profiling worker; runs inside the accepted model runtime environment.

Mirrors the isolation pattern of ``torch_dae.runtime_worker``: this module is executed by a
plain ``python -I -m`` subprocess inside the model's own materialized environment, which has
PyTorch (and the model's vendored/official implementation) installed but intentionally lacks
CodeCarbon/psutil (those are root-only profiling-tooling dependencies; Section 25). It performs
device smoke-testing, architecture profiling, the empirical minimum-input search, and the
canonical latency + resource passes, and returns a single JSON result consumed by
``torch_dae.profiling_executor``.

Only the Python standard library, PyTorch, and the model's own public wrapper are imported here.
"""

from __future__ import annotations

import importlib
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _rss_bytes() -> int:
    """Return the kernel-reported peak RSS so far (``ru_maxrss``), normalized to bytes.

    This is a monotonically non-decreasing high-water mark, not necessarily the exact
    instantaneous RSS at the call site; it requires no additional dependency (``resource`` is
    standard library) and keeps the model-runtime environment free of profiling tooling.
    """

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _import_model_class(entry_point: str) -> Any:
    module_name, symbol = entry_point.split(":")
    return getattr(importlib.import_module(module_name), symbol)


def _torch_device(backend: str, index: int | None) -> Any:
    import torch  # type: ignore[import-not-found]

    label = backend if index is None else f"{backend}:{index}"
    return torch.device(label)


def _configure_threads(thread_regime: str | None) -> dict[str, Any]:
    import torch

    if thread_regime == "single_thread":
        torch.set_num_threads(1)
    info: dict[str, Any] = {"intra_op_threads": torch.get_num_threads()}
    try:
        info["inter_op_threads"] = torch.get_num_interop_threads()
    except RuntimeError:
        info["inter_op_threads"] = None
    return info


def _software_metadata(device: Any) -> dict[str, Any]:
    import torch

    cuda_runtime = cuda_driver = cudnn = mps_info = None
    if device.type == "cuda":
        cuda_runtime = torch.version.cuda
        try:
            cuda_driver = str(torch._C._cuda_getDriverVersion())
        except Exception:  # pragma: no cover - best effort only
            cuda_driver = None
        try:
            cudnn = str(torch.backends.cudnn.version())
        except Exception:  # pragma: no cover - best effort only
            cudnn = None
    if device.type == "mps":
        mps_info = f"pytorch-mps built={torch.backends.mps.is_built()}"
    return {
        "os_name": platform.system(),
        "os_version": platform.release(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": cuda_runtime,
        "cuda_driver_version": cuda_driver,
        "cudnn_version": cudnn,
        "mps_backend_info": mps_info,
    }


def _accelerator_hardware(device: Any) -> dict[str, Any]:
    import torch

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return {
            "accelerator_vendor": "nvidia",
            "accelerator_model": props.name,
            "accelerator_memory_bytes": int(props.total_memory),
        }
    if device.type == "mps":
        return {"accelerator_vendor": "apple", "accelerator_model": "Apple GPU (MPS)"}
    return {}


def _prepare_waveform(*, batch_size: int, channel_count: int, sample_count: int, seed: int) -> Any:
    import numpy as np
    import torch

    from torch_dae.profiling.synthetic_input import generate_white_noise

    array = generate_white_noise(
        seed=seed, batch_size=batch_size, channel_count=channel_count, sample_count=sample_count
    )
    return torch.from_numpy(np.ascontiguousarray(array))


def _try_condition(
    *,
    model: Any,
    device: Any,
    sample_rate: int,
    batch_size: int,
    sample_count: int,
    seed: int,
) -> tuple[bool, str | None]:
    """Construct a waveform and run one forward pass; report acceptance without timing."""

    import torch

    try:
        waveform = _prepare_waveform(
            batch_size=batch_size, channel_count=1, sample_count=sample_count, seed=seed
        ).to(device)
        with torch.no_grad():
            model.forward(waveform, sample_rate)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _run_condition(
    *,
    condition_id: str,
    model: Any,
    device: Any,
    sample_rate: int,
    duration_seconds: float,
    duration_source: str,
    batch_size: int,
    sample_count: int,
    seed: int,
    warmup_count: int,
    measured_count: int,
) -> dict[str, Any]:
    import torch

    from torch_dae.profiling.timing_worker import run_warmup_and_measured

    accepted, reason = _try_condition(
        model=model,
        device=device,
        sample_rate=sample_rate,
        batch_size=batch_size,
        sample_count=sample_count,
        seed=seed,
    )
    if not accepted:
        return {
            "condition_id": condition_id,
            "duration_seconds": duration_seconds,
            "duration_source": duration_source,
            "batch_size": batch_size,
            "sample_count": sample_count,
            "status": "unsupported",
            "unsupported_reason": reason,
            "raw_ns": None,
        }

    waveform = _prepare_waveform(
        batch_size=batch_size, channel_count=1, sample_count=sample_count, seed=seed
    ).to(device)

    def call() -> None:
        with torch.no_grad():
            model.forward(waveform, sample_rate)

    raw_ns = run_warmup_and_measured(
        call, device, warmup_count=warmup_count, measured_count=measured_count
    )
    return {
        "condition_id": condition_id,
        "duration_seconds": duration_seconds,
        "duration_source": duration_source,
        "batch_size": batch_size,
        "sample_count": sample_count,
        "status": "success",
        "unsupported_reason": None,
        "raw_ns": raw_ns,
    }


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    """Run the full worker protocol for one (device, thread-regime) profiling session."""

    import torch

    from torch_dae.profiling.architecture_worker import profile_architecture
    from torch_dae.profiling.device_memory_worker import (
        reset_peak_counters,
        snapshot_accelerator_memory,
    )
    from torch_dae.profiling.minimum_input import search_minimum_supported_samples

    if request["mode"] == "capabilities":
        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }

    torch.manual_seed(request["seed"])
    thread_info = _configure_threads(request.get("thread_regime"))
    device = _torch_device(request["device"]["backend"], request["device"]["index"])

    if request["mode"] == "smoke":
        try:
            model_class = _import_model_class(request["wrapper_entry_point"])
            model = model_class.from_pretrained(checkpoint=Path(request["checkpoint_path"]))
            model = model.to(device).eval()
            ok, reason = _try_condition(
                model=model,
                device=device,
                sample_rate=request["sample_rate"],
                batch_size=1,
                sample_count=int(request["sample_rate"] * 1.0),
                seed=request["seed"],
            )
            return {"smoke_passed": ok, "smoke_error": reason}
        except Exception as exc:
            return {
                "smoke_passed": False,
                "smoke_error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
            }

    rss_before_model_load = _rss_bytes()
    construction_start = time.perf_counter_ns()
    model_class = _import_model_class(request["wrapper_entry_point"])
    model = model_class.from_random()
    construction_end = time.perf_counter_ns()
    model.load_checkpoint(Path(request["checkpoint_path"]))
    checkpoint_loaded_end = time.perf_counter_ns()
    model = model.to(device)
    device_placed_end = time.perf_counter_ns()
    model.eval()
    rss_after_model_load = _rss_bytes()

    architecture = profile_architecture(model).model_dump(mode="json")

    sample_rate = request["sample_rate"]
    canonical_duration = request["canonical_duration_seconds"]
    canonical_source = request["canonical_duration_source"]
    canonical_samples = max(1, round(canonical_duration * sample_rate))

    first_inference_ns: int | None = None
    first_inference_limitations: list[str] = []
    try:
        first_waveform = _prepare_waveform(
            batch_size=1,
            channel_count=1,
            sample_count=canonical_samples,
            seed=request["seed"],
        ).to(device)
        first_start = time.perf_counter_ns()
        with torch.no_grad():
            model.forward(first_waveform, sample_rate)
        first_inference_ns = time.perf_counter_ns() - first_start
    except Exception as exc:
        first_inference_limitations.append(
            f"First-inference cold-start measurement failed: {type(exc).__name__}: {exc}"
        )

    minimum_search = search_minimum_supported_samples(
        accepts=lambda n: _try_condition(
            model=model,
            device=device,
            sample_rate=sample_rate,
            batch_size=1,
            sample_count=n,
            seed=request["seed"],
        )[0],
        sample_rate=sample_rate,
    ).model_dump(mode="json")

    conditions: list[dict[str, Any]] = []
    for batch_size in request["batch_sizes"]:
        conditions.append(
            _run_condition(
                condition_id=f"canonical-b{batch_size}",
                model=model,
                device=device,
                sample_rate=sample_rate,
                duration_seconds=canonical_duration,
                duration_source=canonical_source,
                batch_size=batch_size,
                sample_count=canonical_samples,
                seed=request["seed"],
                warmup_count=request["warmup_count"],
                measured_count=request["measured_count"],
            )
        )

    if (
        minimum_search["status"] == "found"
        and minimum_search["minimum_sample_count"] != canonical_samples
    ):
        min_samples = minimum_search["minimum_sample_count"]
        conditions.append(
            _run_condition(
                condition_id="minimum-b1",
                model=model,
                device=device,
                sample_rate=sample_rate,
                duration_seconds=min_samples / sample_rate,
                duration_source="empirical_minimum",
                batch_size=1,
                sample_count=min_samples,
                seed=request["seed"],
                warmup_count=request["warmup_count"],
                measured_count=request["measured_count"],
            )
        )

    rss_before_resource_pass = _rss_bytes()
    resource_batch = next((c["batch_size"] for c in conditions if c["status"] == "success"), None)
    accelerator_memory = None
    cold_start: dict[str, Any] = {
        "initialization_ns": construction_end - construction_start,
        "checkpoint_loading_ns": checkpoint_loaded_end - construction_end,
        "device_placement_ns": device_placed_end - checkpoint_loaded_end,
        "first_inference_ns": first_inference_ns,
        "limitations": first_inference_limitations,
    }
    if resource_batch is not None:
        reset_peak_counters(device)
        waveform = _prepare_waveform(
            batch_size=resource_batch,
            channel_count=1,
            sample_count=canonical_samples,
            seed=request["seed"],
        ).to(device)
        for _ in range(request["resource_pass_repeats"]):
            with torch.no_grad():
                model.forward(waveform, sample_rate)
        snapshot = snapshot_accelerator_memory(device)
        accelerator_memory = snapshot.model_dump(mode="json") if snapshot is not None else None
    else:
        cold_start["limitations"].append(
            "No condition succeeded; resource pass and accelerator-memory snapshot were skipped."
        )
    rss_after_resource_pass = _rss_bytes()

    return {
        "smoke_passed": True,
        "smoke_error": None,
        "thread_info": thread_info,
        "software": _software_metadata(device),
        "accelerator_hardware": _accelerator_hardware(device),
        "architecture": architecture,
        "cold_start": cold_start,
        "minimum_input_search": minimum_search,
        "conditions": conditions,
        "rss_before_model_load_bytes": rss_before_model_load,
        "rss_after_model_load_bytes": rss_after_model_load,
        "rss_before_resource_pass_bytes": rss_before_resource_pass,
        "rss_after_resource_pass_bytes": rss_after_resource_pass,
        "accelerator_memory": accelerator_memory,
        "canonical_duration_seconds": canonical_duration,
        "canonical_duration_source": canonical_source,
    }


def main() -> int:
    """Run a request produced by ``profiling_executor`` and write the JSON result."""

    request_path, result_path = map(Path, sys.argv[1:])
    try:
        result = execute_request(json.loads(request_path.read_text()))
        result["error"] = None
    except Exception as exc:
        result = {
            "smoke_passed": False,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
        }
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return 0 if result.get("error") is None and result.get("smoke_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
