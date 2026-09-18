"""Pure-torch architecture profiling evidence (Section 18).

Imported only inside a model-environment worker process where PyTorch is guaranteed to be
installed; `torch` is imported lazily inside functions so importing this module elsewhere (e.g.
for type checking) never requires the model runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch_dae.profiling.contracts import ArchitectureEvidence, ArchitectureEvidenceStatus

if TYPE_CHECKING:
    import torch  # type: ignore[import-not-found]


def profile_architecture(model: torch.nn.Module) -> ArchitectureEvidence:
    """Compute mandatory parameter/buffer/dtype evidence for a constructed module.

    FLOPs/MACs are left ``unavailable`` here: no robust, coverage-verified native counting
    backend is wired into Profiling v1, and Section 18 forbids presenting partial operator
    coverage as a complete total. A future backend may populate those fields explicitly with an
    honest ``partial``/``complete`` status.
    """

    total_parameters = 0
    trainable_parameters = 0
    parameter_bytes = 0
    dtype_counts: dict[str, int] = {}
    for parameter in model.parameters():
        count = parameter.numel()
        total_parameters += count
        if parameter.requires_grad:
            trainable_parameters += count
        parameter_bytes += parameter.numel() * parameter.element_size()
        dtype_name = str(parameter.dtype)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + count

    buffer_bytes = 0
    for buffer in model.buffers():
        buffer_bytes += buffer.numel() * buffer.element_size()
        dtype_name = str(buffer.dtype)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + buffer.numel()

    state_dict_tensor_bytes = 0
    for tensor in model.state_dict().values():
        if hasattr(tensor, "numel"):
            state_dict_tensor_bytes += tensor.numel() * tensor.element_size()

    module_count = sum(1 for _ in model.modules())

    return ArchitectureEvidence(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        non_trainable_parameters=total_parameters - trainable_parameters,
        parameter_bytes=parameter_bytes,
        buffer_bytes=buffer_bytes,
        state_dict_tensor_bytes=state_dict_tensor_bytes,
        dtype_distribution=dtype_counts,
        module_count=module_count,
        flops_macs_status=ArchitectureEvidenceStatus.UNAVAILABLE,
    )
