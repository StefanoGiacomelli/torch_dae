# Lifecycle reference

For new onboarding workflows, Model Card lifecycle progression is:

| State | Required meaning |
| --- | --- |
| `draft` | Metadata exists but analysis is incomplete. |
| `analyzed` | Repository, scientific, checkpoint, preprocessing, and embedding evidence was inspected. |
| `environment_resolved` | A locked model-specific environment was constructed and verified. |
| `checkpoint_verified` | The selected checkpoint was acquired, hashed, and loaded compatibly. |
| `runtime_verified` | A passed target-aware report completely covers every target-required runtime check. |

`runtime_verified` is terminal for new onboarding.

The current Model Card schema `1.0.0` still accepts `profiled` and embedded profiling placeholders
for backward compatibility with the original bootstrap contract. New workflows must not use that
state. Profiling v1 produces independent immutable Technical Cards and does not promote or rewrite
the Model Card.

Issues use their own `open`, `resolved`, `accepted`, or `not_applicable` state. Evidence records
distinguish officially reported facts, observations, inferences, unresolved claims, missing reports,
and non-applicable fields.
