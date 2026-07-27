# CLI reference

```text
torch-dae card list
torch-dae card show <card-id>
torch-dae card validate <card-id-or-path>

torch-dae env create <card-id>
torch-dae env ensure <card-id>
torch-dae env verify <card-id>
torch-dae env remove <card-id>
torch-dae env info <card-id>
torch-dae env run <card-id> -- <command>

torch-dae checkpoint ensure <card-id>
torch-dae checkpoint info <card-id>
torch-dae checkpoint remove <card-id>
```

Run `uv run torch-dae <group> --help` for option details. Model inspection and verification CLI
entries are unavailable-feature placeholders in this release.
