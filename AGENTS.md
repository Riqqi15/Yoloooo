# Tactile Navigation Model Context

Read `docs/HANDOFF_TACTILE_V3.md` before changing training data, notebooks, models, evaluation, or mobile integration.

## Non-negotiable rules

- The current model is single-class `tactile_paving` segmentation, not a complete navigation safety system.
- Never train on or tune from protected ground-truth data.
- Never describe a candidate as mobile-ready until every documented gate passes.
- Preserve dataset provenance, manifests, checksums, and the excluded trolley source.
- Keep large binary artifacts in Git LFS.
