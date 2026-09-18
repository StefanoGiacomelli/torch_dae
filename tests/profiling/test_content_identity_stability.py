"""Regression coverage for content-identity stability within one profiling campaign.

Independent review of the first Profiling v1 dogfood run found four different
``torch_dae_content_identity`` values across nine candidate Technical Cards, including three
different identities among three Wavegram cards produced by one campaign. The root cause was
that ``run_profiling_session`` recomputed ``local_package_provenance``/``local_package_identity``
independently for every device/thread condition, so any real edit to the working tree between
those per-device calls (e.g. active development on the profiler itself, dogfooded against
itself) silently produced mixed-provenance cards within a single campaign.

The fix: ``run_campaign`` computes ``LocalPackageProvenance`` exactly once and passes it into
every ``run_profiling_session`` call; ``run_profiling_session`` must never recompute it. These
tests guard that invariant structurally, since exercising the full campaign end-to-end requires
a real model runtime environment and worker subprocess that unit tests cannot stand up.
"""

from __future__ import annotations

import inspect

from torch_dae import profiling_executor


def test_run_profiling_session_does_not_recompute_provenance() -> None:
    """`run_profiling_session` must consume the caller-supplied provenance, not recompute it."""

    source = inspect.getsource(profiling_executor.run_profiling_session)
    assert "local_package_provenance(" not in source
    assert "local_package_identity(" not in source
    assert "content_provenance" in source


def test_run_campaign_computes_provenance_exactly_once_before_the_device_loop() -> None:
    """`run_campaign` must compute provenance once (plus one post-loop consistency check),
    and pass that single value into every device run rather than letting each run derive its
    own."""

    source = inspect.getsource(profiling_executor.run_campaign)
    assert source.count("local_package_provenance(root)") == 2  # start + post-loop verification
    assert "content_provenance=content_provenance" in source


def test_run_profiling_session_requires_content_provenance_argument() -> None:
    signature = inspect.signature(profiling_executor.run_profiling_session)
    assert "content_provenance" in signature.parameters
    assert signature.parameters["content_provenance"].default is inspect.Parameter.empty
