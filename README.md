# Dokumentasi AI Camera Guide

Dokumentasi ini menjadi acuan pembuatan fitur pemandu kamera untuk pengguna tunanetra. Isi dibagi menjadi dua kategori: training model dan implementasi aplikasi.

> **Batas keselamatan:** fitur ini merupakan prototipe bantuan orientasi. Fitur tidak menggantikan tongkat, anjing pemandu, pendamping, atau alat mobilitas tersertifikasi. Saat sistem ragu, terlambat, kehilangan kamera, atau gagal membaca lingkungan, hasil wajib `STOP`; sistem tidak boleh menebak bahwa jalan aman.

## Scope dataset aktif

Fokus: lingkungan stasiun, termasuk peron, area rel, hall, gerbang, ruang tunggu, pintu masuk/keluar, serta trotoar umum yang menjadi akses atau terhubung langsung dengan stasiun. Foto bus/halte sebagai konteks utama dan trotoar tanpa hubungan dengan stasiun tidak masuk dataset aktif.

Daftar foto yang sudah dikurasi tersedia di [data/dataset_scope.json](data/dataset_scope.json). Evaluator menolak foto yang belum terdaftar, entri tanpa file, dan kategori di luar scope sebelum model dimuat. Ini validasi daftar kurasi, bukan pengenalan lokasi otomatis; mengganti isi file dengan nama sama tetap membutuhkan review ulang.

Untuk foto test baru: telaah konteks stasiun, simpan di `data/samples`, lalu tambahkan nama file dan kategori pada `samples` di registry. Trotoar akses memakai `station_access_sidewalk`. Sumber batch lain harus menggunakan registry yang cocok melalui `--scope-file`.

```powershell
rtk .\.venv\Scripts\python.exe scripts\evaluate_samples.py
rtk .\.venv\Scripts\python.exe scripts\render_ground_truth_review.py
rtk .\.venv\Scripts\python.exe scripts\ground_truth_manifest.py validate
rtk .\.venv\Scripts\python.exe scripts\evaluate_ground_truth.py
```

Sebanyak 17 foto saat ini tetap `test_only`: 11 positif tactile dan 6 negatif. Seluruh anotasi telah diperiksa oleh Riyadh dan berstatus `human_reviewed`; evaluator resmi aktif. Validasi format dan metrik offline tetap tidak membuktikan keselamatan penggunaan. Data training baru dikumpulkan terpisah berdasarkan lokasi/sesi.

Intake data training baru berada di [data/training/README.md](data/training/README.md). `scripts/build_training_manifest.py` mewajibkan metadata/provenance, menolak duplikat 17 foto test berdasarkan SHA-256, memeriksa near-duplicate, dan membuat split berbasis lokasi+sesi.

`scripts/training_label_gate.py` memvalidasi box/polygon pada koordinat asli, membuat overlay QA, mengikat persetujuan manusia ke hash anotasi, dan mengekspor YOLO hanya ketika semua sampel `human_reviewed` serta taxonomy berstatus `approved`. Review berubah atau ambiguous membuat export gagal tertutup.

## Kontrak model baseline

Kontrak terpin berada di [models/baseline_manifest.json](models/baseline_manifest.json), dengan ringkasan batas kemampuan di [artifacts/BASELINE_MODEL_CARD.md](artifacts/BASELINE_MODEL_CARD.md). Loader memverifikasi path, ukuran, SHA-256, task, dan raw labels sebelum inferensi.

- Object baseline: YOLO11n detect, 80 kelas COCO. Ini belum mencakup kelas khusus seperti lubang, step-down, tepi peron, atau pintu kaca.
- Tactile baseline: segmentasi satu kelas kandidat tactile. Raw label checkpoint `braille` berbeda dari konfigurasi upstream `tactile_paving`; model belum membedakan guiding dan warning tactile.
- Lisensi Ultralytics dan GuideTWSI memiliki batas berbeda; lihat model card sebelum distribusi.

## Sasaran sistem

- Tanpa jalur taktil: mendeteksi objek dan bahaya yang berada dalam koridor berjalan di depan kamera.
- Dengan jalur taktil: mengikuti mask jalur, tetap mendeteksi seluruh objek, dan menghentikan pengguna jika jalur tertutup, berakhir, atau menuju perubahan permukaan berbahaya.
- Mendeteksi orang, kursi, meja, tiang, tempat sampah, tas, pembatas, tangga, lubang, kenaikan, penurunan, dan tepi peron sesuai kemampuan model serta sensor.
- Menggabungkan YOLO object detection, YOLO segmentation jalur taktil, dan depth jika perangkat mendukung.
- Menjalankan keputusan keselamatan secara on-device tanpa bergantung pada Gemini, backend, atau internet.
- Menargetkan latency persepsi-ke-keputusan di bawah 200 ms setelah warm-up pada perangkat target.
- Menyimpan model sebagai asset aplikasi. Frame kamera tidak disimpan di Neon.

## Peta dokumentasi

Acuan status implementasi dan urutan perbaikan: [Audit mismatch dan roadmap](docs/AUDIT_MISMATCH_DAN_ROADMAP.md). Dokumen rancangan di bawah mencakup fitur yang sebagian masih belum diimplementasikan.

### Kategori 1 — Training

1. [Dataset dan labeling](training/01-dataset-and-labeling.md)
2. [Training Colab dan export model](training/02-colab-training-and-export.md)
3. [Evaluasi dan keselamatan model](training/03-evaluation-and-model-safety.md)

### Kategori 2 — Implementasi

1. [Arsitektur runtime](implementation/01-runtime-architecture.md)
2. [Functional safety](implementation/02-functional-safety.md)
3. [Kompatibilitas perangkat dan performa](implementation/03-device-compatibility-and-performance.md)
4. [Demo, pengujian, dan observability](implementation/04-demo-testing-and-observability.md)
5. [Proyek serupa, tools, dan improvisasi](implementation/05-related-projects-tools-and-improvements.md)

## Baseline yang direkomendasikan

| Kebutuhan | Baseline |
|---|---|
| Objek umum | YOLO11n detect, pretrained COCO, lalu fine-tune data ROD dan data lokal |
| Jalur taktil | GuideTWSI `yolo11n_tactile.pt`, lalu fine-tune foto lokal |
| Jarak | ARCore Depth pada perangkat yang mendukung |
| Perangkat tanpa ARCore | kategori dekat/sedang/jauh; jangan mengucapkan jarak meter |
| Runtime Flutter | plugin resmi `ultralytics_yolo` atau native LiteRT bridge jika hasil benchmark plugin tidak cukup |
| Feedback | TTS singkat, pola getaran, dan bunyi STOP prioritas tertinggi |
| Safety | watchdog independen, hasil stale dibuang, fail-safe stop |

## Urutan pengerjaan

1. Selesaikan dataset lokal dan aturan label.
2. Latih serta validasi object detector.
3. Fine-tune segmenter jalur taktil.
4. Export dua model nano menjadi LiteRT/TFLite terkuantisasi.
5. Integrasikan object-only mode.
6. Tambahkan tactile-path mode dan fusion mask dengan objek.
7. Tambahkan watchdog dan seluruh fail-safe sebelum depth atau arahan berjalan diaktifkan.
8. Tambahkan ARCore sebagai kemampuan opsional.
9. Jalankan benchmark perangkat dan pilih capability profile otomatis.
10. Jalankan demo terkontrol, replay test, dan review risk register.

## Referensi utama

- [Google Research Project Guideline](https://github.com/google-research/project-guideline)
- [GuideTWSI dataset dan pretrained weights](https://github.com/DARoSLab/GuideTWSI)
- [ROD obstacle dataset](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics Flutter plugin](https://pub.dev/packages/ultralytics_yolo)
- [ARCore Depth](https://developers.google.com/ar/develop/depth)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

Referensi model baseline dan hash artefak diperiksa pada 5 September 2026. Lisensi setiap dataset, model, dan library wajib diperiksa kembali sebelum distribusi atau penggunaan komersial.

## Workspace pengembangan lokal

Kode training dan utilitas AI dikerjakan langsung dalam folder ini. Virtual environment memakai Python 3.11 dan PyTorch CPU karena laptop tidak memiliki GPU NVIDIA.

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\check_environment.py
```

Struktur runtime lokal:

```text
AI Camera/
├── .venv/          # dependency lokal, tidak dibagikan
├── data/           # dataset lokal, tidak masuk Git
├── models/         # checkpoint/model download
├── runs/           # output training/evaluasi
├── artifacts/      # model final dan model card
├── notebooks/      # notebook Colab/VS Code
└── scripts/        # preprocessing, training helper, dan verifikasi
```

Training berat tetap dijalankan di Google Colab. Laptop dipakai untuk coding, validasi dataset, smoke test CPU, dan integrasi hasil model.
