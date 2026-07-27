Model-card contracts
====================

A :class:`ModelCard` serializes exactly one model-family, variant, and checkpoint tuple. All models
are strict Pydantic contracts: unknown fields are rejected, JSON uses the declared field names, and
validation failures are reported as ``pydantic.ValidationError``. Repository paths must be relative,
must not contain ``..``, and are resolved only by the consuming manager.

Lifecycle
---------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Exact meaning and promotion requirements
   * - ``draft``
     - Contract is structurally valid; facts, environment, checkpoint, and runtime evidence may
       remain unresolved.
   * - ``analyzed``
     - Static/scientific analysis is recorded. No additional runtime invariant is imposed by the
       model validator.
   * - ``environment_resolved``
     - ``usage.recommended_environment.verified`` must be true and its committed references must be
       repository-relative.
   * - ``checkpoint_verified``
     - Environment requirements above plus ``checkpoint.observed_sha256``.
   * - ``runtime_verified``
     - Checkpoint requirements above plus a repository-relative ``verification_report``.
   * - ``profiled``
     - Runtime requirements above plus architectural, inference, and energy profiling sections all
       marked ``profiled``, each with a report path.

The lifecycle is monotonic as a documentation convention, but the contract validates the required
state rather than performing promotion or reading referenced files.

Other enums
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Enum
     - Value
     - Meaning
   * - ``EvidenceStatus``
     - ``officially_reported``
     - Explicit primary upstream statement.
   * -
     - ``observed``
     - Locally visible source or runtime observation.
   * -
     - ``inferred``
     - Reasoned conclusion; ``rationale`` is required.
   * -
     - ``unresolved``
     - Evidence is insufficient or ambiguous.
   * -
     - ``not_reported``
     - Relevant upstream information was not reported.
   * -
     - ``not_applicable``
     - Field does not apply.
   * - ``IssueStatus``
     - ``open`` / ``resolved`` / ``accepted`` / ``not_applicable``
     - Unresolved work, completed resolution, knowingly accepted issue, or inapplicable issue.
   * - ``ProfilingStatus``
     - ``not_profiled`` / ``profiled``
     - No completed report, or a completed report whose repository-relative path is required.

Field contract matrix
---------------------

Every listed field is public and serialized. A field is required unless a default is shown.

.. list-table::
   :header-rows: 1
   :widths: 18 47 35

   * - Contract
     - Fields
     - Principal invariants/defaults
   * - ``EvidenceRecord``
     - ``evidence_id``, ``kind``, ``status``, ``url?``, ``revision?``, ``path?``, ``symbol?``,
       ``description``, ``rationale?``
     - Path is repository-relative; inferred status requires rationale.
   * - ``IssueRecord``
     - ``issue_id``, ``kind``, ``status``, ``description``, ``impact``
     - Canonical identifier and exact enum status.
   * - ``Identity``
     - ``model_name``, ``model_family``, ``variant``, ``checkpoint_name``, ``framework``,
       ``wrapper_entry_point``
     - Framework is ``pytorch``; entry point is ``module:CapitalizedClass``.
   * - ``SourceRecord``
     - ``source_id``, ``kind``, ``url?``, ``package?``, ``revision?``, ``path?``,
       ``evidence_status``
     - Revision format is constrained; path is repository-relative.
   * - ``Sources``
     - ``official_repository``, ``implementation``, ``checkpoint``, ``wrapper``
     - Each role is a ``SourceRecord``.
   * - ``ScientificReference``
     - ``title``, ``doi?``, ``official_publication?``, ``authors``, ``year=None``
     - Year, when present, is 1900–2100.
   * - ``Description``
     - ``architecture``, ``preprocessing``, ``training_objective``, ``checkpoint_behavior``,
       ``implementation``
     - Claims remain separated by topic.
   * - ``Tasks``
     - ``pretraining``, ``finetuning``, ``official_evaluation``, ``supported_inference``
     - Each is a tuple of task descriptions.
   * - ``DatasetRecord``
     - ``name``, ``version=None``, ``subset=None``, ``split=None``, ``role``, ``source_status``,
       ``evidence_ids=()``
     - Evidence references must resolve at card level.
   * - ``Datasets``
     - ``training``, ``validation``, ``testing``
     - Explicit partitions; empty tuples are allowed.
   * - ``MetricRecord``
     - ``task``, ``dataset``, ``split``, ``metric``, ``value``, ``unit=None``, ``protocol``,
       ``checkpoint_specific``, ``source_status``, ``evidence_ids=()``
     - Unit is explicit when applicable; evidence resolves at card level.
   * - ``RecommendedEnvironment``
     - ``environment_id``, ``specification``, ``lockfile``, ``verified``
     - Both paths are repository-relative.
   * - ``Usage``
     - ``recommended_environment``, ``installation_commands``, ``checkpoint_loading``,
       ``smoke_test_command``
     - Commands are documentation; validation does not execute them.
   * - ``WaveformInput``
     - ``shape``, ``sample_rate_hz``, ``dtype="float32"``, ``valid_lengths_shape="B"``,
       ``channels``, ``resampling``, ``padding``, ``normalization``
     - Shape is exactly ``B,C,T``; sample rate is positive hertz; valid lengths are ``B`` or null.
   * - ``OutputComponent``
     - ``name``, ``kind``, ``semantic_kind``, ``rank``, ``layout``, ``dimensions``, ``dtype?``,
       ``granularity?``, ``task_head_relation?``
     - Nonnegative rank equals the number of dimension names.
   * - ``Outputs``
     - ``primary``, ``components``, ``probability_output=None``
     - Probability presence must agree with the card capability.
   * - ``EmbeddingsSection``
     - ``default_embedding_id``, ``items``
     - IDs are unique; exactly one item is default and its ID matches ``default_embedding_id``.
   * - ``BooleanCapability``
     - ``supported``, ``reason=None``
     - Unsupported capabilities require a reason.
   * - ``CapabilitiesSection``
     - ``random_initialization``, ``checkpoint_loading``, ``probabilities``, ``embeddings``
     - Probability and embedding flags agree with declared outputs at card level.
   * - ``DeviceSupport``
     - ``upstream_declared``, ``locally_tested``, ``known_limitations``
     - Declared and observed support stay distinct.
   * - ``ProfilingSection``
     - ``status``, ``report=None``
     - ``profiled`` requires a repository-relative report.

``ModelCard`` additionally requires ``schema_version``, ``card_id``, ``card_status``, ``identity``,
``checkpoint``, ``sources``, ``scientific_reference``, ``description``, ``tasks``, ``datasets``,
``reported_metrics``, ``usage``, ``input``, ``outputs``, ``embeddings``, ``capabilities``,
``device_support``, ``verification_report=None``, the three profiling sections, ``limitations``,
``issues``, and ``evidence``. Evidence IDs are unique; every dataset, metric, and embedding reference
must resolve. License metadata remains informational. Validation imports no wrapper, downloads no
checkpoint, and creates no environment.

.. autoclass:: torch_dae.cards.models.ModelCardLifecycle

.. autoclass:: torch_dae.cards.models.EvidenceStatus

.. autoclass:: torch_dae.cards.models.IssueStatus

.. autoclass:: torch_dae.cards.models.ProfilingStatus

.. autoclass:: torch_dae.cards.models.EvidenceRecord

.. autoclass:: torch_dae.cards.models.IssueRecord

.. autoclass:: torch_dae.cards.models.Identity

.. autoclass:: torch_dae.cards.models.SourceRecord

.. autoclass:: torch_dae.cards.models.Sources

.. autoclass:: torch_dae.cards.models.ScientificReference

.. autoclass:: torch_dae.cards.models.Description

.. autoclass:: torch_dae.cards.models.Tasks

.. autoclass:: torch_dae.cards.models.DatasetRecord

.. autoclass:: torch_dae.cards.models.Datasets

.. autoclass:: torch_dae.cards.models.MetricRecord

.. autoclass:: torch_dae.cards.models.RecommendedEnvironment

.. autoclass:: torch_dae.cards.models.Usage

.. autoclass:: torch_dae.cards.models.WaveformInput

.. autoclass:: torch_dae.cards.models.OutputComponent

.. autoclass:: torch_dae.cards.models.Outputs

.. autoclass:: torch_dae.cards.models.EmbeddingsSection

.. autoclass:: torch_dae.cards.models.BooleanCapability

.. autoclass:: torch_dae.cards.models.CapabilitiesSection

.. autoclass:: torch_dae.cards.models.DeviceSupport

.. autoclass:: torch_dae.cards.models.ProfilingSection

.. autoclass:: torch_dae.cards.models.ModelCard
