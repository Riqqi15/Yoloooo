# Colab Training Handoff Design

## Goal

Make the tactile-paving training work reproducible for a collaborator from a fresh Google Colab runtime, with every required input hosted in `Riqqi15/Yoloooo` and an accurate project handoff that Codex can discover immediately.

## Target model saat ini

The current trainable artifact is a **single-class instance-segmentation model** for `tactile_paving`. Its immediate job is to produce a stable corridor mask that a later navigation decision layer can use. It is not yet a detector for people, holes, platform edges, train doors, or distance.

A candidate may continue to mobile export only when all of these gates pass:

- protected/local-test mask recall is at least `0.90`;
- protected/local-test positive mean IoU is at least `0.75`;
- protected/local-test false-positive rate does not exceed the current baseline of `0.50`;
- it beats the baseline model without a class mismatch;
- 20 repeated controlled trials produce no critical miss in each scenario: straight tactile corridor, tactile corridor partially blocked by a person or bag, left/right turn, and yellow non-tactile hard negative;
- the candidate bundle is complete and known failure cases are recorded in its model card.

The next training target is `tactile-one-class-v3-public2-station1`. It must be trained from the GuideTWSI checkpoint, compared with the completed 4:1 candidate, then tested through the protected gate above.

## Target sistem akhir

The eventual camera application should guide a blind user along tactile paving, announce a stable left or right turn, detect people or obstacles crossing the walking corridor, and issue `STOP` for holes, drop-offs, or a platform edge near railway tracks. Five metres is the target observation range when supported by depth calibration; it is not a guaranteed safe distance from a monocular segmentation model.

The controlled demonstration goal ends at entering a Commuter train carriage. Crossing the platform-train gap must require explicit user confirmation or human supervision. Hazard alerts always override route guidance, and missing, stale, or ambiguous perception must result in `STOP` or `Uncertain`.

This final goal requires separate people/obstacle and hazard perception, depth or distance estimation, temporal stability, and a navigation decision layer. Passing the current tactile-model gate is only the first component milestone and must not be presented as proof that the complete safety system is ready.

## Current truth

- The `public4-station1` candidate completed 80 epochs in about 28 minutes on a Colab T4.
- Training validation reported box precision `0.9005`, box recall `0.7354`, box mAP50 `0.8378`, box mAP50-95 `0.6984`, mask precision `0.8958`, mask recall `0.7385`, mask mAP50 `0.8250`, and mask mAP50-95 `0.6979`.
- Protected evaluation at confidence `0.05` produced TP `11`, FN `0`, FP `6`, TN `0`, recall `1.0`, precision `0.6471`, positive mean IoU `0.5641`, false-positive rate `1.0`, and median CPU latency `92.1 ms`.
- The protected gate rejected the 4:1 candidate because mean IoU missed the `0.75` target and all 6 negative images produced false positives. The high training mAP is not equivalent to deployment accuracy.
- The local `public2-station1` run was stopped during epoch 5 and is not a completed candidate. A collaborator must train this candidate from the GuideTWSI checkpoint.
- The public cache contains 3,960 files in a 329.33 MiB ZIP and is licensed CC0/Public Domain.

### Score sementara yang ditampilkan dalam handoff

The handoff must show the completed 4:1 candidate as follows, without collapsing unlike metrics into a single accuracy percentage:

| Evaluation | Metric | Score sementara | Target | Status |
| --- | --- | ---: | ---: | --- |
| Training validation | Mask mAP50 | `82.50%` | Informational | Recorded |
| Training validation | Mask mAP50-95 | `69.79%` | Informational | Recorded |
| Protected test | Mask recall | `100.00%` | `>= 90%` | Pass |
| Protected test | Positive mean IoU | `56.41%` | `>= 75%` | Fail |
| Protected test | Precision | `64.71%` | Informational | Recorded |
| Protected test | False-positive rate | `100.00%` | `<= 50%` | Fail |

Overall temporary result: **rejected / not mobile-ready**. There is no valid single “accuracy score” because training mAP, protected recall, overlap quality, and false-positive rate measure different behavior. The next candidate must improve overlap quality and negative-scene rejection without losing recall.

## Improvement roadmap

The handoff must preserve this order. Later-stage work must not hide an unresolved failure in an earlier stage.

### P0 — complete the next controlled comparison

1. Train `tactile-one-class-v3-public2-station1` from the GuideTWSI checkpoint with the committed seed, split, Ultralytics version, image size, and training configuration.
2. Validate the resulting bundle before describing the run as complete. Record the commit, dataset version, manifest checksum, checkpoint checksum, seed, GPU, elapsed time, epoch count, and all box/mask metrics.
3. Select the confidence threshold on the ordinary validation split only, freeze it, and run the protected test once. Never tune hyperparameters or labels from protected-test images.
4. Compare the 2:1 candidate with the completed 4:1 candidate and the baseline. Select by the full protected gate, not by training mAP alone.
5. Keep a downloadable `last.pt` or an optional Google Drive copy so a Colab interruption can resume with the same configuration instead of restarting. A partial checkpoint remains marked incomplete.

### P0 — reduce the observed errors

- Prioritize false-positive reduction because the current protected false-positive rate is `100%`. Add legal, newly sourced hard negatives covering ordinary yellow floor tiles, painted lines, platform markings, floor seams, shadows, rails, platform edges, and other repeating textures that resemble tactile paving.
- Improve mask boundaries because protected positive mean IoU is only `56.41%`. Audit polygon consistency for thin or distant paths, turns, intersections, partial visibility, perspective narrowing, occlusion, low light, blur, glare, and cropped paths.
- Do not copy protected-test images or labels into training. Collect visually analogous training examples instead.
- Preserve the user's exclusion of the trolley source image; do not silently restore it. Obstacles such as trolleys belong in a separately reviewed obstacle dataset later.
- Use realistic augmentations only. Avoid transforms that destroy tactile geometry or create impossible platform scenes.

### P1 — strengthen data and evaluation quality

- Expand the independent station validation coverage; the current 4:1 manifest has only 13 station validation samples, which is too small to represent all conditions reliably.
- Split by station, source group, or capture sequence rather than by individual frame alone, and audit hashes and near-duplicates to prevent closely related scenes leaking across train, validation, and protected test.
- Add scenario tags and report metrics separately for straight paths, turns, intersections, distant paths, occlusion, low light, blur, glare, non-tactile yellow surfaces, platform edges, and crowded scenes.
- Expand the protected set beyond its current 17 images before making any safety claim. Keep it immutable and inaccessible to the training notebook.
- After the main 2:1 experiment, repeat the winning configuration with additional seeds when GPU quota allows. Promote it only if the result is stable rather than a lucky single run.
- Store qualitative overlays for failures and review whether each error comes from the model, an inconsistent label, or an ambiguous scene before changing the dataset.
- Track both image-level false-positive rate and false-positive region count so one image containing many incorrect masks is not understated.

### P1 — validate navigation behaviour on video

- Convert the tactile mask into a walkable corridor only after the segmentation gate passes.
- Require at least three consecutive confident frames before announcing a left or right turn; otherwise continue straight, say `Uncertain`, or issue `STOP` as appropriate.
- Test 20 repetitions per controlled scenario and record critical misses, unstable direction changes, time-to-alert, and recovery after brief occlusion.
- Test the 5-metre observation target using measured distances, multiple camera heights and angles, day/night lighting, motion blur, and representative phone cameras. Do not infer a guaranteed distance from pixel area alone.

### P2 — add the missing safety components

- Add a separate people/obstacle detector and intersect its detections with the tactile corridor. A person or object outside the corridor should not trigger the same warning as one blocking the route.
- Build separately reviewed classes or models for holes/drop-offs, platform edge/track danger, train door, and platform-train gap. Hazard output always overrides route guidance.
- Add calibrated depth estimation or a depth-capable sensor before giving metre-based proximity warnings.
- Require explicit confirmation or human supervision before crossing the platform-train gap in the controlled Commuter demonstration.
- Design fail-safe audio and vibration states: `Straight`, `Left`, `Right`, `Obstacle`, `STOP`, and `Uncertain`. Missing frames, stale inference, low confidence, or model failure must never default to “safe”.

### P2 — optimize only after accuracy gates pass

- Keep YOLO11n segmentation as the first mobile-sized candidate. Try a larger segmentation model or higher input resolution only if data and label improvements still cannot reach the gate, and record the latency trade-off.
- Benchmark the exported model on the actual target phone. The current `92.1 ms` median CPU measurement is provisional and includes slow outliers; it is not a mobile real-time guarantee.
- Measure end-to-end camera latency, thermal throttling, memory use, battery draw, and speech/haptic delay, not just neural-network inference time.
- Re-run mask agreement and protected safety gates after FP32, FP16, or INT8 conversion. A successful export is not proof of equivalent behaviour.

## Repository layout

- `AGENTS.md`: short, automatically discovered Codex context and a pointer to the full handoff.
- `docs/HANDOFF_TACTILE_V3.md`: project overview, verified progress, artifact inventory, exact continuation checklist, and safety constraints.
- `data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip`: immutable public cache bundle stored with Git LFS.
- `data/public/guidetwsi-rbar-v1/sha256.txt`: checksum for the cache bundle.
- `notebooks/train_tactile_v3_public_colab.ipynb`: clean, output-free notebook that trains only the unfinished 2:1 candidate by default.

## Colab data flow

1. Install the pinned Ultralytics version and Git LFS when absent.
2. Clone `codex/model-first-mobile-ready` with LFS smudging disabled.
3. Pull only the initial checkpoint, station dataset, completed 4:1 candidate metadata/model, and public cache ZIP required for this workflow.
4. Verify the ZIP SHA-256 and its 3,960-file payload before extraction.
5. Build `tactile-v3-public2-station1` from the committed public manifest and station dataset.
6. Require an active CUDA GPU and train `tactile-one-class-v3-public2-station1` for up to 80 epochs with existing early stopping.
7. Validate the candidate bundle and download it as a ZIP.

No protected ground-truth image or label is read during training. Protected evaluation remains a separate local gate after the collaborator returns the candidate ZIP.

## Consistency rules

- The public ZIP is treated as immutable. Replacing it requires updating `sha256.txt`, provenance, manifest, and the documented file count together.
- The notebook contains no saved outputs or ad-hoc diagnostic cells.
- The notebook never retrains the completed 4:1 candidate unless a maintainer deliberately changes `TRAIN_RATIOS`.
- An interrupted run directory is not represented as a completed candidate.
- A candidate is never described as mobile-ready until protected evaluation passes.

## Verification

- Static notebook contract tests cover the Git LFS include list, checksum verification, file count, selected ratio, GPU requirement, packaging, and absence of protected-data paths.
- Existing dataset-builder and training-configuration tests continue to pass.
- A clean-clone smoke test confirms every referenced tracked file and Git LFS object exists.
- The pushed branch and LFS upload are verified against `origin/codex/model-first-mobile-ready`.

## Out of scope

- Mobile export and application integration.
- Training people, obstacle, hole, platform-edge, or train-door detectors.
- Declaring the one-class tactile model safe based only on training metrics.
