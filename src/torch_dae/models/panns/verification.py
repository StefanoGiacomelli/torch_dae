"""Bounded checkpoint observations using the integrated PANNs public wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]

from torch_dae.core.errors import UnsupportedCapabilityError
from torch_dae.environment.verification import TensorDimension, TensorObservation
from torch_dae.models.panns.labels import load_audioset_labels
from torch_dae.models.panns.model import (
    _MINIMUM_SAMPLE_COUNTS,
    _load_panns_checkpoint_payload,
)
from torch_dae.runtime_verification import RuntimeVerificationTarget


class Provider:
    """PANNs observations behind the generic managed-worker protocol."""

    def __init__(self, model_class: Any, target: RuntimeVerificationTarget, checkpoint: Path):
        self.target = target
        self.checkpoint = checkpoint
        self.model_class = model_class
        self.model: Any = None
        self.loaded = False
        self.payload: Any = None
        self.output: Any = None
        self.tensor_observations: list[TensorObservation] = []
        self.embedding_results: list[str] = []
        self.rate = target.required_sample_rate_hz
        self.samples = self.rate * 64 // 100
        if self.samples > target.verification_limits.maximum_samples_per_item:
            raise ValueError("target sample limit cannot accommodate canonical input")
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.waveform = torch.linspace(-0.5, 0.5, self.samples).reshape(1, 1, -1)

    def check(self, name: str) -> str:
        name = {
            "random-initialization": "random_initialization",
            "checkpoint-deserialization": "checkpoint_deserialization",
            "checkpoint-strict-state-dictionary-load": "checkpoint_strict_load",
            "invalid-checkpoint-rejected": "invalid_checkpoint",
            "integrated-variant-agreement": "variant_agreement",
            "canonical-waveform-input": "cpu_canonical_forward",
            "batch-behavior": "cpu_batch_forward",
            "zero-input-behavior": "zero_input",
            "duration-constraints": "duration_contract",
            "equal-valid-lengths": "valid_lengths_contract",
            "invalid-input-rejection": "input_validation_contract",
            "checkpoint-loaded-cpu-forward": "cpu_loaded_forward",
            "checkpoint-output-contract": "output_contract",
            "checkpoint-probability-contract": "probability_contract",
            "checkpoint-native-output-parity": "native_output_parity",
            "checkpoint-default-embedding": "default_embedding_contract",
            "audioset-label-contract": "audioset_label_contract",
            "checkpoint-deterministic-evaluation": "eval_determinism",
            "cpu-device-placement": "cpu_device_transfer",
            "checkpoint-differentiability": "cpu_differentiability",
            "automatic-resampling": "automatic_resampling",
            "padded-variable-valid-lengths": "padded_variable_lengths",
            "mps-native-execution": "mps_forward",
        }.get(name, name)
        method = getattr(self, "check_" + name, None)
        if method is None:
            raise UnsupportedCapabilityError(f"No PANNs observation implements {name}.")
        if (
            name
            not in {
                "checkpoint_deserialization",
                "checkpoint_payload_contract",
                "random_initialization",
                "automatic_resampling",
                "padded_variable_lengths",
                "checkpoint_strict_load",
            }
            and not self.loaded
        ):
            raise RuntimeError("checkpoint strict loading did not pass; forward is not authorized")
        return str(method())

    def check_checkpoint_deserialization(self) -> str:
        if self.target.checkpoint.loader not in {
            "payload['model'] followed by load_state_dict(state_dict, strict=True)",
            (
                "torch.load(path, map_location=device)['model'] "
                "followed by strict model.load_state_dict"
            ),
        }:
            raise ValueError("unsupported PANNs checkpoint loader contract")
        self.payload = _load_panns_checkpoint_payload(self.checkpoint, map_location="cpu")
        self.check_checkpoint_payload_contract()
        return "torch.load(weights_only=True, map_location='cpu'): model state mapping validated."

    def check_checkpoint_payload_contract(self) -> str:
        assert isinstance(self.payload, Mapping) and "model" in self.payload
        assert isinstance(self.payload["model"], Mapping)
        return f"payload['model'] is a mapping with {len(self.payload['model'])} state entries."

    def check_checkpoint_strict_load(self) -> str:
        self.payload = None
        identity = self.model_class.identity
        assert identity.model_id == self.target.adapter_id == self.target.registry_identity
        assert identity.variant_id == self.target.integrated_variant_id
        assert identity.model_id == self.target.checkpoint.checkpoint_id
        assert identity.sample_rate == self.rate
        if self.model is None:
            self.model = self.model_class.from_random(
                variant=self.target.integrated_variant_id
            ).eval()
        self.model.load_checkpoint(self.checkpoint, strict=True, map_location="cpu")
        self.loaded = True
        return "Public load_checkpoint(strict=True): all keys matched; no missing/unexpected keys."

    def forward(self, waveform: Any, **kwargs: Any) -> Any:
        assert waveform.shape[0] <= self.target.verification_limits.maximum_batch_size
        assert waveform.shape[-1] <= self.target.verification_limits.maximum_samples_per_item
        with torch.no_grad():
            result = self.model(waveform, self.rate, **kwargs)
        for tensor in result.tensors.values():
            assert torch.isfinite(tensor).all(), "nonfinite runtime output"
        return result

    def check_cpu_canonical_forward(self) -> str:
        self.output = self.forward(self.waveform)
        return f"Checkpoint-loaded float32 [1,1,{self.samples}] forward; all tensors finite."

    def check_cpu_batch_forward(self) -> str:
        assert self.target.verification_limits.maximum_batch_size >= 2
        batch = torch.cat((self.waveform, self.waveform * 0.5), dim=0)
        output = self.forward(batch)
        assert output.primary.shape == (2, 527)
        torch.testing.assert_close(output.primary[:1], self.output.primary, rtol=1e-4, atol=1e-5)
        return "Two-item batch: finite [2,527]; first item agrees with single forward."

    def check_zero_input(self) -> str:
        output = self.forward(torch.zeros_like(self.waveform))
        assert output.primary.shape == (1, 527)
        return "Zero waveform produces finite logits, probabilities and embedding."

    def check_output_contract(self) -> str:
        for expected in self.target.expected_outputs:
            tensor = self.output.tensors[expected.name]
            shape = tuple(1 if d == "B" else int(d) for d in expected.dimensions)
            assert tuple(tensor.shape) == shape and tensor.ndim == expected.rank
            assert tensor.dtype == torch.float32 and tensor.device.type == "cpu"
            assert torch.isfinite(tensor).all()
            self.tensor_observations.append(
                TensorObservation(
                    name=expected.name,
                    role="embedding" if expected.name == "embedding" else "output",
                    component_path=f"outputs.tensors.{expected.name}",
                    rank=tensor.ndim,
                    shape=tuple(
                        TensorDimension(name=d, size=n, dynamic=d == "B")
                        for d, n in zip(expected.dimensions, tensor.shape, strict=True)
                    ),
                    dtype="float32",
                    device="cpu",
                )
            )
        return "Declared output ranks, shapes, float32 dtype, CPU placement and finiteness match."

    def check_probability_contract(self) -> str:
        assert self.target.probability_semantics == "sigmoid"
        probabilities = self.output.tensors["probabilities"]
        assert torch.all((probabilities >= 0) & (probabilities <= 1))
        torch.testing.assert_close(
            torch.sigmoid(self.output.primary), probabilities, rtol=1e-6, atol=1e-7
        )
        with torch.no_grad():
            predicted = self.model.predict_probability(self.waveform, self.rate)
        torch.testing.assert_close(predicted, probabilities, rtol=0, atol=0)
        return "[1,527] public predict_probability equals sigmoid(logits), bounded in [0,1]."

    def check_native_output_parity(self) -> str:
        prepared = self.model.preprocess(self.waveform, self.rate)
        with torch.no_grad():
            native = self.model.upstream_model(prepared.model_input)
        for public, upstream in (
            ("logits", "logits"),
            ("probabilities", "clipwise_output"),
            ("embedding", "embedding"),
        ):
            torch.testing.assert_close(
                self.output.tensors[public], native[upstream], rtol=0, atol=0
            )
        return "Public tensors equal a separate native upstream forward exactly in eval mode."

    def check_default_embedding_contract(self) -> str:
        expected = self.target.default_embedding
        assert self.model.default_embedding_id == expected.embedding_id
        specifications = self.model.available_embeddings()
        assert len(specifications) == 1 and specifications[0].dimension == expected.dimension
        with torch.no_grad():
            embedding = self.model.compute_embedding(self.waveform, self.rate)
        assert embedding.tensor.shape == (1, expected.dimension)
        torch.testing.assert_close(
            embedding.tensor, self.output.tensors["embedding"], rtol=0, atol=0
        )
        self.embedding_results.append(embedding.embedding_id)
        return f"Default embedding: [1,{expected.dimension}], equal to native embedding."

    def check_audioset_label_contract(self) -> str:
        labels = load_audioset_labels()
        assert len(labels) == self.output.primary.shape[1] == 527
        assert self.model.class_labels == tuple(item.display_name for item in labels)
        return "527 output columns match upstream AudioSet index/MID/display-name order."

    def check_eval_determinism(self) -> str:
        for _ in range(self.target.verification_limits.repeated_calls):
            repeated = self.forward(self.waveform)
            for name, value in self.output.tensors.items():
                torch.testing.assert_close(repeated.tensors[name], value, rtol=0, atol=0)
        return "All logits, probabilities and embeddings repeat exactly with eval/dropout disabled."

    @staticmethod
    def rejected(call: Callable[[], Any], exception: type[Exception]) -> None:
        try:
            call()
        except exception:
            return
        raise AssertionError(f"expected {exception.__name__}")

    def check_input_validation_contract(self) -> str:
        cases = (
            (self.waveform.squeeze(1), self.rate, ValueError),
            (self.waveform.expand(1, 2, -1), self.rate, ValueError),
            (self.waveform.to(torch.int16), self.rate, TypeError),
            (self.waveform, self.rate + 1, ValueError),
        )
        for waveform, rate, exception in cases:
            self.rejected(
                lambda w=waveform, r=rate: self.model.preprocess(w, r, allow_resample=False),  # type: ignore[misc]
                exception,
            )
        return "Wrong rank, stereo, integer dtype and sample rate rejected."

    def check_duration_contract(self) -> str:
        minimum = _MINIMUM_SAMPLE_COUNTS[self.target.adapter_id]
        self.rejected(
            lambda: self.model.preprocess(torch.zeros(1, 1, minimum - 1), self.rate), ValueError
        )
        for samples in (self.samples, self.rate, self.rate * 2):
            output = self.forward(torch.zeros(1, 1, samples))
            assert output.primary.shape == (1, 527)
        if self.target.adapter_id == "panns-wavegram-logmel-cnn14-map-0439":
            self.rejected(
                lambda: self.model.preprocess(torch.zeros(1, 1, 10560), self.rate), ValueError
            )
        return f"Below {minimum} samples rejected; 0.64s/1s/2s zeros passed; alignment enforced."

    def check_valid_lengths_contract(self) -> str:
        batch = self.waveform.expand(2, 1, -1).contiguous()
        baseline = self.forward(batch)
        output = self.forward(batch, valid_lengths=torch.tensor([self.samples, self.samples]))
        torch.testing.assert_close(output.primary, baseline.primary, rtol=0, atol=0)
        self.rejected(
            lambda: self.model.preprocess(
                batch, self.rate, valid_lengths=torch.tensor([self.samples])
            ),
            ValueError,
        )
        return "Equal [B] valid_lengths preserves output; wrong vector shape rejected."

    def check_cpu_device_transfer(self) -> str:
        self.model.to("cpu")
        assert all(
            t.device.type == "cpu" for t in (*self.model.parameters(), *self.model.buffers())
        )
        self.forward(self.waveform)
        return "Public .to('cpu') parameters, buffers and checkpoint-loaded forward remain on CPU."

    def check_cpu_differentiability(self) -> str:
        self.model.zero_grad(set_to_none=True)
        waveform = self.waveform.clone().requires_grad_(True)
        output = self.model(waveform, self.rate)
        (output.primary.square().mean() + output.tensors["embedding"].square().mean()).backward()
        assert waveform.grad is not None, "waveform gradient is absent"
        assert torch.isfinite(waveform.grad).all(), "waveform gradient contains NaN or Inf"
        nonzero_input_gradients = int(torch.count_nonzero(waveform.grad))
        classifier_parameters = [
            (name, parameter)
            for name, parameter in self.model.upstream_model.named_parameters()
            if name.startswith("fc_audioset.") and parameter.requires_grad
        ]
        assert classifier_parameters, "trainable fc_audioset parameters are absent"
        for name, parameter in classifier_parameters:
            assert parameter.grad is not None, f"classifier gradient absent: {name}"
            assert torch.isfinite(parameter.grad).all(), (
                f"classifier gradient contains NaN or Inf: {name}"
            )
        backbone_parameter_name = None
        for name, parameter in self.model.upstream_model.named_parameters():
            if name.startswith("fc_audioset.") or not parameter.requires_grad:
                continue
            if parameter.grad is not None and torch.isfinite(parameter.grad).all():
                backbone_parameter_name = name
                break
        assert backbone_parameter_name is not None, (
            "no trainable non-classifier parameter has a finite gradient"
        )
        self.model.zero_grad(set_to_none=True)
        return (
            "Logits/embedding backpropagate finite waveform, all trainable fc_audioset "
            f"gradients and backbone gradient {backbone_parameter_name}; "
            f"waveform gradient nonzero entries={nonzero_input_gradients}/{self.samples}. "
            "Zero gradients on this input do not imply a disconnected graph."
        )

    def check_automatic_resampling(self) -> str:
        try:
            self.model.preprocess(self.waveform, self.rate // 2, allow_resample=True)
        except ValueError as exc:
            if "automatic resampling is not supported" in str(exc):
                raise UnsupportedCapabilityError(str(exc)) from exc
            raise
        return "Automatic resampling accepted by public preprocessing."

    def check_padded_variable_lengths(self) -> str:
        batch = self.waveform.expand(2, 1, -1).contiguous()
        self.forward(batch, valid_lengths=torch.tensor([self.samples, self.samples - 1]))
        return "Padded unequal valid lengths accepted."

    def check_mps_forward(self) -> str:
        if "mps" not in self.target.permitted_devices or not torch.backends.mps.is_available():
            raise UnsupportedCapabilityError("Native MPS is unavailable on the execution host.")
        try:
            self.model.to("mps")
            with torch.no_grad():
                output = self.model(self.waveform.to("mps"), self.rate)
            assert all(
                t.device.type == "mps" and torch.isfinite(t).all() for t in output.tensors.values()
            )
        except NotImplementedError as exc:
            raise UnsupportedCapabilityError(f"Native MPS operator unsupported: {exc}") from exc
        finally:
            self.model.to("cpu")
        return "Native MPS checkpoint-loaded forward passed with fallback disabled."

    def _mps_observation(self, operation: str) -> str:
        if "mps" not in self.target.permitted_devices or not torch.backends.mps.is_available():
            raise UnsupportedCapabilityError("Native MPS is unavailable on the execution host.")
        try:
            self.model.to("mps")
            assert all(t.device.type == "mps" for t in self.model.parameters())
            if operation == "transfer":
                return "Public .to('mps') parameter placement passed."
            waveform = self.waveform.to("mps")
            if operation == "differentiability":
                waveform.requires_grad_(True)
                self.model(waveform, self.rate).primary.sum().backward()
                assert waveform.grad is not None and torch.isfinite(waveform.grad).all()
            else:
                with torch.no_grad():
                    first = self.model(waveform, self.rate)
                    for _ in range(self.target.verification_limits.repeated_calls):
                        second = self.model(waveform, self.rate)
                        torch.testing.assert_close(first.primary, second.primary, rtol=0, atol=0)
        except NotImplementedError as exc:
            raise UnsupportedCapabilityError(f"Native MPS operator unsupported: {exc}") from exc
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.to("cpu")
        return f"Native MPS {operation} passed with fallback disabled."

    def check_mps_device_transfer(self) -> str:
        return self._mps_observation("transfer")

    def check_mps_determinism(self) -> str:
        return self._mps_observation("determinism")

    def check_mps_differentiability(self) -> str:
        return self._mps_observation("differentiability")

    def check_random_initialization(self) -> str:
        self.model = self.model_class.from_random(variant=self.target.integrated_variant_id).eval()
        output = self.forward(self.waveform)
        assert output.primary.shape == (1, 527)
        return "Public from_random constructor and finite synthetic CPU forward passed."

    def check_invalid_checkpoint(self) -> str:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pth"
            torch.save({"model": {}}, path)
            self.rejected(lambda: self.model.load_checkpoint(path, strict=True), RuntimeError)
            torch.save({"wrong-key": {}}, path)
            self.rejected(lambda: self.model.load_checkpoint(path, strict=True), ValueError)
        return "Public strict loader rejects empty state-dictionary and missing model payload key."

    def check_variant_agreement(self) -> str:
        assert self.model.identity.variant_id == self.target.integrated_variant_id
        assert self.model.identity.model_id == self.target.checkpoint.checkpoint_id
        self.rejected(lambda: self.model_class.from_random(variant="wrong-variant"), ValueError)
        return "Fixed wrapper/checkpoint/variant identities agree; wrong variant rejected."

    def check_cpu_loaded_forward(self) -> str:
        assert self.loaded
        output = self.forward(self.waveform)
        assert output.primary.device.type == "cpu" and output.primary.shape == (1, 527)
        return "Checkpoint-loaded public CPU forward completed with finite outputs."
