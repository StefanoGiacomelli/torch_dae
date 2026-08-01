# Lifecycle And Decision Gates

Use the committed lifecycle states from `project_spec.md`: `draft`, `analyzed`,
`environment_resolved`, `checkpoint_verified`, `runtime_verified`, and `profiled`.

Separately track environment states `draft`, `materialized`, and `environment_verified`. These
belong to the environment and cannot promote a model/checkpoint tuple. Runtime verification belongs
to the explicit model/checkpoint/environment target. A final card consumes both evidence layers.
Only an `EnvironmentVerificationResult` with `verification_status=passed` may establish
`environment_verified`, and it must completely cover its required import and smoke observations with
passed status. Only a target-aware `VerificationReport` with successful overall status and complete
passed coverage of its target-required checks may support `runtime_verified`. Failed, unsupported,
missing, duplicate, undeclared, or legacy evidence never promotes either lifecycle.

Decision gates are required when multiple scientifically meaningful alternatives remain: card scope,
variant, checkpoint, source implementation, source strategy, embedding, preprocessing, output
semantics, technical access/authentication blockers, license metadata issues,
distribution/publication decisions, or wrapper equivalence.

Do not ask for decisions that objective evidence can resolve.

Control-plane drift across accepted onboarding phases is not a lifecycle transition or a decision
gate. Accepted hashes remain historical provenance, while every pending phase candidate must use the
current skill and specification hashes. A later phase may consume older accepted prerequisites and
must expose that transition without rewriting them or relaxing current contracts.
