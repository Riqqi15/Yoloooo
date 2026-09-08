# Tactile Ground-Truth Review

`manifest.csv` contains test-only images. Model prediction columns are hints, not truth. Importing an annotation produces `provisional`; it never grants human-review status.

For every row:

1. Generate and inspect source-resolution QA overlays:

```powershell
rtk .\.venv\Scripts\python.exe scripts\render_ground_truth_review.py
```

Open files in `runs/ground-truth-review/`. Red areas are ground-truth polygons, not model predictions.

2. Inspect the original image and correct its Labelme JSON when the polygon is shifted, includes an occluder/watermark, or misses visible tactile pixels.
3. Set `ground_truth_tactile_present` to `1` when tactile paving is visible, otherwise `0`.
4. For a positive row, create the PNG named by `ground_truth_mask_path`:
   - same width and height as source image;
   - single-channel/grayscale PNG;
   - background value `0`;
   - tactile-paving pixels value `255`.
5. For a negative row, leave mask absent or provide an all-zero mask.
6. Keep uncertain images as `ambiguous`. Do not force them to positive or negative.
7. After a real person has inspected the source and overlay, record that explicit action with the `review` command below.

Do not copy predicted masks as ground truth. Draw masks by inspecting the source image.

## Labelme workflow

1. Open images from `data/samples/` in Labelme.
2. Draw polygon shapes using exactly this label: `tactile_paving`.
3. Save JSON files into `data/ground_truth/labelme/`.
4. For images without tactile paving, save a JSON with no shapes.
5. Import all available JSON files:

```powershell
rtk .\.venv\Scripts\python.exe scripts\import_labelme_ground_truth.py
```

Importer refuses unknown labels, wrong dimensions, invalid polygons, and previously annotated rows. New masks are written to a versioned batch; the manifest switches only after the whole batch validates. Default provenance is `ai_assisted`. Use `--annotation-origin human` only when polygons were actually authored by a person. `--overwrite` replaces an earlier annotation with a new provisional version; it does not preserve human approval.

Mark an uncertain draft without approving it:

```powershell
rtk .\.venv\Scripts\python.exe scripts\ground_truth_manifest.py ambiguous --samples "sample.jpg" --reason "jalur tertutup dan tekstur tidak dapat dipastikan"
```

Record a completed human review:

```powershell
rtk .\.venv\Scripts\python.exe scripts\ground_truth_manifest.py review --reviewer "nama-reviewer" --samples "sample.jpg"
```

The command records a timestamp. Run it only after inspecting that sample. It accepts provisional rows; an ambiguous sample must first be corrected and reimported.

Validate:

```powershell
rtk .\.venv\Scripts\python.exe scripts\ground_truth_manifest.py validate
```

Exit `0` means all rows are valid and human-reviewed. Exit `2` means provisional/ambiguous/incomplete work remains. Exit `1` means annotation data is invalid.

Official evaluation rejects provisional and ambiguous rows:

```powershell
rtk .\.venv\Scripts\python.exe scripts\evaluate_ground_truth.py
```

Diagnostic evaluation may explicitly accept provisional rows, but still rejects ambiguous rows and writes to a separate file:

```powershell
rtk .\.venv\Scripts\python.exe scripts\evaluate_ground_truth.py --allow-provisional
```
