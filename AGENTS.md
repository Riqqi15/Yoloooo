# Tactile Navigation Model Context

## Mandatory Codex onboarding

This file is the repository instruction entrypoint. Codex must stop before making any change and read [`CODEX_START_HERE.md`](CODEX_START_HERE.md). That file is the required reading manifest for the project and points to the handoff, score baseline, training plan, and safety constraints.

At minimum, read these documents before changing training data, notebooks, models, evaluation, or mobile integration:

1. [`CODEX_START_HERE.md`](CODEX_START_HERE.md)
2. [`docs/HANDOFF_TACTILE_V3.md`](docs/HANDOFF_TACTILE_V3.md)
3. [`README.md`](README.md)

If a task changes the Colab workflow, also read [`docs/superpowers/plans/2026-09-10-colab-training-handoff.md`](docs/superpowers/plans/2026-09-10-colab-training-handoff.md) and [`docs/superpowers/specs/2026-09-10-colab-training-handoff-design.md`](docs/superpowers/specs/2026-09-10-colab-training-handoff-design.md). Do not claim that the instructions were followed without reading the applicable documents.

## Non-negotiable rules

- The current model is single-class `tactile_paving` segmentation, not a complete navigation safety system.
- Never train on or tune from protected ground-truth data.
- Never describe a candidate as mobile-ready until every documented gate passes.
- Preserve dataset provenance, manifests, checksums, and the excluded trolley source.
- Keep large binary artifacts in Git LFS.
