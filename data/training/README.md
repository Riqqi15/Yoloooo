# Intake Dataset Training Stasiun

Folder ini terpisah dari 17 foto `data/samples` yang berperan `test_only`.

## Alur

1. Taruh foto baru di `inbox/`.
2. Salin `intake_template.csv` menjadi `intake.csv`, lalu isi satu row per foto.
3. Pastikan `usage_permission` menjelaskan dasar penggunaan, bukan sekadar `yes` tanpa catatan sumber.
4. Jalankan:

```powershell
rtk .\.venv\Scripts\python.exe scripts\build_training_manifest.py --dataset-version station-v1
```

Script menolak konteks di luar stasiun, file rusak/tidak didukung, metadata kosong, exact duplicate, konten yang sama dengan 17 test aktif, serta near-duplicate yang jatuh ke split berbeda. Split ditentukan dari gabungan `location_id` dan `session_id`, bukan frame acak.

`near_duplicate_of` adalah kandidat berbasis dHash dan tetap perlu pemeriksaan manusia. `annotation_status` selalu dimulai sebagai `unreviewed`.

## ID nonidentitas

- `location_id`: kode stabil seperti `station_a_platform_1`; hindari alamat rinci bila tidak dibutuhkan.
- `session_id`: satu sesi pengambilan berurutan. Frame dari sesi yang sama selalu masuk split yang sama.
- `source_name`: asal dataset/kamera/pengambil data tanpa identitas pribadi.
- `usage_permission`: jenis izin atau lisensi dan rujukannya.

Taxonomy objek/dua-kelas lama tetap berupa draft di `taxonomy_v1.json`. Untuk pipeline tactile satu kelas yang telah disetujui Riyadh, gunakan `taxonomy_tactile_v1.json` dengan label `tactile_paving`.

## Gate label dan export

Setelah manifest intake dan anotasi tersedia:

```powershell
rtk .\.venv\Scripts\python.exe scripts\training_label_gate.py --manifest data\training\manifests\station-v1.json --taxonomy data\training\taxonomy_tactile_v1.json render
rtk .\.venv\Scripts\python.exe scripts\training_label_gate.py --manifest data\training\manifests\station-v1.json --taxonomy data\training\taxonomy_tactile_v1.json review --reviewer NAMA --samples SAMPLE_ID
rtk .\.venv\Scripts\python.exe scripts\training_label_gate.py --manifest data\training\manifests\station-v1.json --taxonomy data\training\taxonomy_tactile_v1.json validate
rtk .\.venv\Scripts\python.exe scripts\training_label_gate.py --manifest data\training\manifests\station-v1.json --taxonomy data\training\taxonomy_tactile_v1.json export --output-dir artifacts\datasets\station-v1
```

Jangan menjalankan `review` sebelum manusia melihat overlay. Export menolak taxonomy draft, missing/ambiguous review, hash berubah, geometri rusak, kelas asing, dan konten yang sama dengan test lama.

## Training tactile satu kelas

Upload folder export `artifacts/datasets/station-v1` dan checkpoint GuideTWSI ke Google Colab, lalu jalankan `notebooks/train_tactile_one_class_colab.ipynb` dengan runtime GPU. Notebook mem-pin Ultralytics 8.4.138, memvalidasi split dan label, menjalankan smoke inference, fine-tuning, validation, serta membuat bundle candidate.

Setelah bundle diunduh ke workspace, verifikasi:

```powershell
rtk .\.venv\Scripts\python.exe scripts\verify_tactile_candidate.py artifacts\candidates\tactile-one-class-v1
```

Status `candidate_valid` hanya membuktikan kelengkapan dan integritas artefak; bukan izin deployment.
