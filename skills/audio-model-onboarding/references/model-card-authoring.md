# Model Card Authoring

Cards are checkpoint-specific: one model family, one variant, one checkpoint. Every populated field
must trace to upstream source, paper or official docs, environment evidence, checkpoint evidence,
runtime observation, or explicit user decision.

Use nulls, unresolved states, TODO markers where allowed, and explicit issues when evidence is
absent. Validate committed cards through both Pydantic and generated JSON Schema.
