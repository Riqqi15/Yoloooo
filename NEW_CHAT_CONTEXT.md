# Handoff AI Camera — Status Aktif

Tanggal: 8 September 2026.

## Batas kerja

- Workspace: `C:\Users\riyadh\Downloads\AI Camera`.
- Jangan membuka atau mengubah `C:\Users\riyadh\Timetable-scheduling`.
- Scope dataset: lingkungan stasiun dan trotoar yang menjadi akses langsung ke stasiun.
- Bus/halte sebagai konteks utama, jalan umum, dan trotoar tanpa hubungan stasiun dikecualikan.
- Tujuh belas foto aktif adalah `test_only`; jangan masukkan ke train/validation atau pakai untuk memilih hyperparameter.
- Semua hasil runtime wajib fail-safe STOP. Proyek belum memberi izin berjalan dan bukan alat mobilitas tersertifikasi.

## Sumber kebenaran

1. `docs/AUDIT_MISMATCH_DAN_ROADMAP.md`
2. `README.md`
3. `models/baseline_manifest.json`
4. `artifacts/BASELINE_MODEL_CARD.md`
5. `data/dataset_scope.json`

File di `docs/superpowers/plans` dan `docs/superpowers/specs` adalah catatan milestone historis.

## Progres terverifikasi

### Baseline dan scope

- Object inference: `scripts/baseline_object_inference.py`, YOLO11n detect COCO.
- Tactile inference: `scripts/baseline_tactile_inference.py`, GuideTWSI segment satu kelas.
- Batch: `scripts/evaluate_samples.py`; registry scope diperiksa sebelum model dimuat.
- Format WEBP didukung.
- Batch terakhir: 17/17 foto berhasil, semua status STOP; prediction masks dan report berada di `runs/sample-evaluation/`.

### Kontrak model

- Manifest terpin: `models/baseline_manifest.json`.
- Object SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Tactile SHA-256: `79f0bb39a354dd66d367bb5b97184df626c85f91568456e099e49a8cdbfcd878`.
- Loader menolak path, size, hash, task, atau raw labels yang berubah sebelum inferensi.
- Tactile raw label `{0: braille}` berbeda dari konfigurasi upstream `{0: tactile_paving}`. Output hanya kandidat tactile satu kelas; belum membedakan guiding vs warning.
- Object model hanya 80 kelas COCO. Lubang, step, tepi peron, pintu kaca, dan beberapa bahaya khusus stasiun belum tersedia sebagai kelas.

### Ground truth dan evaluasi

- Manifest mencatat hash sumber/mask, provenance, annotation ID, status, reviewer, dan waktu review.
- Import Labelme memakai batch mask versioned dan rollback atomik.
- Overlay QA tersedia di `runs/ground-truth-review/`.
- Status aktif: 17 `human_reviewed` oleh Riyadh, 0 provisional, 0 ambiguous; 11 positif dan 6 negatif.
- Evaluator resmi lulus. Hasil aktif: accuracy 70,59%, precision 75%, recall 81,82%, F1 78,26%, dan mean IoU 0,5318 pada 11 gambar GT positif.
- Laporan resmi berada di `runs/accuracy/tactile_metrics.json`; metrik lama tetap diarsipkan di `runs/accuracy/archive/`.

### Verifikasi terakhir

- 57 unit tests lulus pada 7 September 2026.
- Python byte-compilation lulus.
- Batch aktual 17/17 lulus dengan manifest model aktif.

### Pipeline training tactile satu kelas

- Riyadh menyetujui taxonomy `data/training/taxonomy_tactile_v1.json` dengan satu label `tactile_paving`.
- Notebook Colab reproducible tersedia di `notebooks/train_tactile_one_class_colab.ipynb` dan mem-pin Ultralytics 8.4.138.
- `scripts/verify_tactile_candidate.py` memvalidasi bundle candidate, checksum, task segmentation, label, dan metadata training.
- Dataset `station-photo-set-50-v1` sudah lolos gate: 50/50 `human_reviewed`, 13 foto positif, 37 negatif, dan 30 polygon `tactile_paving`.
- Export YOLO aktif berada di `artifacts/datasets/station-photo-set-50-v1` dengan split 35 train, 8 validation, dan 7 test. Masing-masing split memiliki sampel positif.

## Pekerjaan berikut

1. Upload `artifacts/datasets/station-photo-set-50-v1` dan checkpoint awal ke Google Colab.
2. Jalankan `notebooks/train_tactile_one_class_colab.ipynb` dengan runtime GPU.
3. Unduh bundle candidate dan validasi dengan `scripts/verify_tactile_candidate.py`.
4. Evaluasi kandidat pada holdout/failure set sebelum mempertimbangkan export mobile.
5. Runtime Android tetap tahap berikut dan belum diimplementasikan.

## Perintah utama

```powershell
cd "C:\Users\riyadh\Downloads\AI Camera"
rtk .\.venv\Scripts\python.exe scripts\evaluate_samples.py
rtk .\.venv\Scripts\python.exe scripts\render_ground_truth_review.py
rtk .\.venv\Scripts\python.exe scripts\ground_truth_manifest.py validate
rtk .\.venv\Scripts\python.exe scripts\evaluate_ground_truth.py
rtk .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Jangan memasang ulang dependency bila pemeriksaan environment berhasil.
