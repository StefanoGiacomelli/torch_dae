# Analyze mode

`analyze` inventories an upstream repository without importing or executing its code. It records
repository identity, scientific claims, architecture candidates, dependencies, preprocessing,
outputs, checkpoint candidates, embedding candidates, source strategies, open questions, and
confidence counts.

The result is a machine-readable analysis report plus deterministic Markdown. It does not create a
model environment, download a checkpoint, or add a wrapper.

Binary files encountered during checkpoint discovery are reported and skipped per path without
discarding valid textual candidates. Output candidates carry lexical class/method ownership and
source spans. Claims and embeddings can be scoped to declared variant/checkpoint IDs, and
host-published checksums retain their exact algorithms without claiming local verification.

Local PDFs supplied as scientific or technical references may be read with the bounded
`extract_pdf_text.py` utility. It extracts no OCR and its output remains reading material until
reviewed and cited under the evidence policy.

Checkpoint/source chronology is best-effort. Prefer explicit host revision metadata, an associated
signed or annotated tag, or an authoritative cited commit. Repository-history dating and the nearest
preceding commit are inference only. Current default-branch HEAD is current-source evidence and is
not automatically checkpoint-equivalent.

With `WORKFLOW_ID`, accepted output is promoted to
`onboarding_reports/<workflow-id>/analyze/` before completion, followed by the canonical `finalize`
command. The response reports the handoff, review bundle digest, cleanup result, and retained
managed caches.
