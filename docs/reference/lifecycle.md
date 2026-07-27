# Lifecycle reference

| State | Required meaning |
| --- | --- |
| `draft` | Metadata exists but analysis is incomplete. |
| `analyzed` | Repository, scientific, checkpoint, preprocessing, and embedding evidence was inspected. |
| `environment_resolved` | A locked model-specific environment was constructed and verified. |
| `checkpoint_verified` | The selected checkpoint was acquired, hashed, and loaded compatibly. |
| `runtime_verified` | Public inputs, outputs, embeddings, devices, and declared gradients were verified. |
| `profiled` | The separate profiling protocol completed. |

Issues use their own `open`, `resolved`, `accepted`, or `not_applicable` state. Evidence records
distinguish officially reported facts, observations, inferences, unresolved claims, missing reports,
and non-applicable fields.

The validator's exact promotion prerequisites—including verified environment, observed checkpoint
hash, runtime report, and three completed profiling reports—are documented with
{class}`torch_dae.cards.ModelCard` in {doc}`../api/model-cards`. A lifecycle label records satisfied
contract conditions; it does not run verification or resolve an open issue.
