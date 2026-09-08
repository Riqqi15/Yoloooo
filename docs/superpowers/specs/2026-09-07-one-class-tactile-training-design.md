# Desain Training Tactile Satu Kelas

Tanggal: 7 September 2026

Status: disetujui Riyadh pada 7 September 2026

## Tujuan

Menyediakan pipeline Google Colab yang reproducible untuk fine-tuning checkpoint GuideTWSI menjadi segmenter lokal satu kelas bernama `tactile_paving`.

Pipeline hanya menerima dataset training baru yang telah lolos intake, validasi anotasi, dan review manusia. Sebanyak 17 foto di `data/samples` tetap `test_only` dan tidak boleh dipakai untuk training, validation, tuning, atau pemilihan checkpoint.

## Keputusan utama

- Gunakan satu kelas `tactile_paving`; guiding path dan warning block belum dipisahkan.
- Mulai dari `models/guidetwsi/yolo11n_tactile.pt`.
- Training dijalankan di Google Colab GPU, bukan CPU laptop.
- Gunakan Ultralytics versi terpin dan seed tetap.
- Pilih checkpoint serta threshold hanya dari validation set.
- Gunakan 17 foto test aktif hanya setelah konfigurasi dikunci.
- Training objek khusus stasiun berada di luar cakupan pekerjaan ini.

## Input

Pipeline menerima artefak dataset YOLO hasil `scripts/training_label_gate.py export`. Struktur wajib:

```text
dataset/
├── tactile/
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── data.yaml
└── export_manifest.json
```

`tactile/data.yaml` harus berisi tepat satu kelas: `tactile_paving`. `export_manifest.json` menjadi bukti versi dataset dan sampel yang diekspor. Taxonomy training baru mempertahankan daftar objek yang ada agar kompatibel dengan gate, tetapi tactile diubah menjadi satu kelas dan statusnya `approved` berdasarkan pilihan pemilik proyek. Pipeline ini hanya memakai bagian tactile.

## Komponen

### Notebook Colab

Satu notebook menjalankan alur berikut secara berurutan:

1. Memeriksa GPU dan memasang versi dependency terpin.
2. Memuat dataset export dan checkpoint awal.
3. Memvalidasi file wajib, taxonomy satu kelas, split train/validation, dan ketiadaan hash 17 test aktif.
4. Menjalankan smoke test singkat untuk memastikan dataset dan checkpoint dapat dimuat.
5. Menjalankan fine-tuning YOLO segmentation dengan seed tetap.
6. Menjalankan validation pada `best.pt`.
7. Menyimpan checkpoint, argumen training, versi dependency, metrik, dan checksum SHA-256.

Notebook tidak mengunduh atau menyusun dataset secara diam-diam. Dataset harus disiapkan melalui gate lokal terlebih dahulu.

### Verifikasi artefak

Helper lokal kecil memeriksa paket hasil Colab:

- `best.pt` tersedia dan bukan file kosong;
- task model adalah segmentation;
- raw label tepat `{0: tactile_paving}`;
- metrik, konfigurasi training, versi dependency, dan checksum tersedia;
- checksum yang dicatat cocok dengan checkpoint.

Helper tidak menentukan bahwa model aman. Ia hanya memvalidasi kelengkapan dan integritas artefak.

## Konfigurasi awal

Konfigurasi baseline menggunakan `imgsz=640`, `epochs=80`, `seed=42`, GPU tunggal, dan batch otomatis. Augmentasi dibatasi pada perubahan cahaya, blur ringan, perspektif kecil, dan occlusion ringan yang tidak mengubah makna fisik tactile.

Nilai ini menjadi baseline, bukan hasil tuning. Percobaan berikutnya hanya mengubah satu kelompok parameter dan dicatat sebagai run terpisah.

## Tuning

Tuning menggunakan validation set. Parameter awal yang boleh dibandingkan:

- ukuran input 512 dan 640;
- learning rate default Ultralytics dan satu nilai lebih kecil;
- augmentasi ringan aktif atau nonaktif;
- confidence threshold berdasarkan precision-recall validation.

Test set aktif tidak dibuka untuk memilih parameter. Setelah satu konfigurasi dipilih, evaluasi test dijalankan dan hasilnya dicatat tanpa mengulang tuning terhadap test.

## Kegagalan dan fail-closed

Pipeline berhenti jika dataset kosong, taxonomy bukan satu kelas, split hilang, checkpoint salah task, hash test ditemukan, dependency tidak sesuai, training gagal, atau artefak hasil tidak lengkap.

Kegagalan tidak menghasilkan status model kandidat. Checkpoint parsial tidak dipindahkan ke folder artefak final.

## Output

```text
artifacts/candidates/tactile-<run-id>/
├── best.pt
├── metrics.json
├── training_config.json
├── environment.txt
├── sha256.txt
└── MODEL_CARD.md
```

Output berstatus `candidate`, bukan model aplikasi final. Export LiteRT dan integrasi Android dilakukan setelah evaluasi serta parity check terpisah.

## Verifikasi implementasi

- Unit test helper artefak memakai checkpoint/metadata dummy dan mencakup hash salah, label salah, task salah, serta file hilang.
- Notebook diuji melalui pemeriksaan struktur dan eksekusi smoke cell; training penuh membutuhkan Colab GPU serta dataset training nyata.
- Seluruh test suite lokal harus tetap lulus.

## Batas saat ini

Pipeline dapat dibangun sebelum data training tersedia, tetapi training nyata tetap tertahan sampai foto baru, metadata, polygon, dan review manusia lengkap. Hasil test 17 foto saat ini tidak membuktikan keselamatan berjalan atau kesiapan deployment.
