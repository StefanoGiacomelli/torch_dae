Static onboarding inspection
============================

Inspection is bounded and static. It inventories files, parses TOML/JSON/configuration and Python
ASTs, and searches text for explicit candidates. It never imports inspected modules, runs
``setup.py``, executes an upstream helper, installs dependencies, fetches a URL, or opens model
weights.

Traversal excludes version-control, virtual-environment, runtime, build, cache, and dependency
directories. Supported artifact symlinks are rejected, external symlinks are recorded rather than
followed, and all evidence paths must remain inside the inspected root. Defaults limit a single file
to 512,000 bytes, the operation to 10,000 files, and charged content to 5,000,000 bytes. One shared
:class:`InspectionBudget` prevents nested inspectors from resetting those limits; repeated reads use
its in-memory text cache.

Inspector responsibilities
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Function
     - Returned observations and limits
   * - ``inspect_repository``
     - Deterministic inventory, file kinds, directories, licenses, dependencies, archives, and
       checkpoint-like paths.
   * - ``inspect_python_project``
     - Static packaging metadata, requirements, supported lock/environment files, source roots, and
       source-strategy assessment.
   * - ``inspect_dependencies``
     - Normalized declarations, exact/constraint evidence, AST imports, unpinned dependencies, and
       narrow recognized API risks.
   * - ``inspect_imports``
     - Module names and attribute references from ASTs; no import execution.
   * - ``inspect_model_candidates``
     - ``nn.Module``-like class and model/load/preprocess-named function candidates.
   * - ``inspect_output_candidates``
     - Forward return keys and tensor-like assignment names, without claiming embedding semantics.
   * - ``inspect_checkpoints``
     - Local checkpoint-like paths, literal URLs/hashes, and statically resolvable helper
       associations; unresolved components remain explicit.
   * - ``classify_source_strategy``
     - Packaging, framework, revision, officiality, vendoring, external-implementation, ambiguity,
       and decision evidence. It does not choose among unresolved strategies.
   * - ``generate_environment_candidates``
     - Evidence-ranked, untried compatibility candidates with versions, risks, uncertainties,
       source identity, and decision gates.
   * - ``inspect_scenario_repository``
     - One validated aggregate built by the same production inspectors under one shared budget.

Malformed files, unsafe paths, unsupported symlinks, syntax errors, and budget exhaustion raise
:class:`OnboardingInspectionError`. Missing or ambiguous evidence usually remains in returned
candidates rather than becoming a fact.

Synthetic offline example
-------------------------

.. code-block:: python

   from pathlib import Path
   from tempfile import TemporaryDirectory
   from torch_dae.onboarding import InspectionBudget, inspect_imports

   with TemporaryDirectory() as directory:
       root = Path(directory)
       (root / "model.py").write_text("import torch\nclass Encoder: pass\n")
       result = inspect_imports(root, budget=InspectionBudget(maximum_total_files=4))
       assert result["frameworks"] == ["torch"]

The write creates only the caller's temporary fixture. Inspection itself is read-only and does not
import ``torch``.

.. autoexception:: torch_dae.onboarding.inspection.OnboardingInspectionError

.. autoclass:: torch_dae.onboarding.inspection.InspectionBudget

   .. automethod:: visit_file
   .. automethod:: cached_text
   .. automethod:: store_text

.. autofunction:: torch_dae.onboarding.inspection.inspect_repository

.. autofunction:: torch_dae.onboarding.inspection.inspect_python_project

.. autofunction:: torch_dae.onboarding.inspection.inspect_dependencies

.. autofunction:: torch_dae.onboarding.inspection.inspect_imports

.. autofunction:: torch_dae.onboarding.inspection.inspect_model_candidates

.. autofunction:: torch_dae.onboarding.inspection.inspect_output_candidates

.. autofunction:: torch_dae.onboarding.inspection.inspect_checkpoints

.. autofunction:: torch_dae.onboarding.inspection.generate_environment_candidates

.. autofunction:: torch_dae.onboarding.inspection.classify_source_strategy

.. autofunction:: torch_dae.onboarding.inspection.inspect_scenario_repository
