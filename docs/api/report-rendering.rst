Report rendering
================

:func:`render_analysis_markdown` accepts a validated :class:`~torch_dae.onboarding.AnalysisReport`
and returns a deterministic Markdown view. It renders identity and sections in a fixed order,
preserves tuple order for candidates, questions, decisions, and evidence, and spells out confidence
counts and the recommended next mode. Missing optional identity values render as ``unresolved``;
empty candidate collections render as ``none``.

Canonical JSON remains the machine contract. Markdown is a presentation derived from it and is not
parsed back into state. The renderer does not fetch sources, infer absent values, reorder evidence,
or invent facts.

A minimal fragment:

.. code-block:: python

   from torch_dae.onboarding import AnalysisReport, render_analysis_markdown

   report = AnalysisReport.model_validate_json(canonical_json)
   markdown = render_analysis_markdown(report)
   assert markdown.startswith(f"# Technical Analysis Report: {report.report_id}")

Here ``canonical_json`` denotes a complete validated report string; constructing that large contract
is covered in :doc:`onboarding-contracts`.

.. autofunction:: torch_dae.onboarding.rendering.render_analysis_markdown
