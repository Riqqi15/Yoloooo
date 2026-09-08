# Station Dataset Scope Gate Design

## Goal

Keep the active dataset limited to station environments while allowing sidewalks that directly serve, connect to, or visibly belong to a station.

## Scope

Included contexts are station interior, concourse, platform, track area, station entrance or exit, and adjacent station-access sidewalk. Bus-only, bus-stop-only, generic road, and unrelated public sidewalk scenes stay outside the active dataset.

## Design

`data/dataset_scope.json` is the explicit allowlist. Every active image filename maps to one approved station context. `scripts/evaluate_samples.py` loads the registry before model loading and fails safe if an active image is unregistered, a registered image is missing, or a context is not approved. The report records the scope and per-image context for auditability.

This uses an allowlist instead of an automatic scene classifier because semantic filtering errors would silently contaminate the dataset. New station images require one deliberate registry entry.

## Verification

Unit tests cover accepted station and station-sidewalk contexts, unregistered active images, missing registered images, invalid scope names, and invalid contexts. The real ten-image batch must pass the gate and contain no bus record.

## Non-goals

- No automatic semantic scene classification.
- No training on the existing `test_only` images.
- No deletion of excluded source files.
