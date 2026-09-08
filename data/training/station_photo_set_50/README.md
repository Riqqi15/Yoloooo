# Station photo set — 50 foto

Set kurasi untuk review manual. Berisi 5 foto yang sudah dipilih sebelumnya + 45 foto Wikimedia Commons baru. Semua foto menampilkan stasiun/peron/interior/jalur stasiun; foto langit-dominan, non-stasiun, close-up sinyal/marka, kereta-only, dan duplikat tidak dimasukkan.

Status: **lolos gate dan sudah diekspor untuk training** pada 8 September 2026. Seluruh 50 foto berstatus `human_reviewed` oleh Riyadh: 13 foto positif dengan 30 polygon `tactile_paving` dan 37 foto negatif. Split berbasis grup lokasi/sesi memakai seed 1: 35 train, 8 validation, dan 7 test.

`sources.json` menyimpan halaman sumber, URL unduh, fotografer, dan lisensi bila tersedia. Untuk foto Wikimedia, pertahankan kredit fotografer + tautan halaman sumber + tautan lisensi CC BY-SA sesuai metadata sumber. Foto CC0 tidak membutuhkan atribusi, tetapi sumber tetap dicatat.

Kandidat yang tidak dipilih tetap ada di `../station_review_batch/`; tidak dihapus.

Artefak aktif:

- Manifest: `../manifests/station-photo-set-50-v1.json`
- Anotasi gate: `../annotations/station-photo-set-50-v1/`
- Laporan QA: `../../../runs/training-label-review/station-photo-set-50-v1/`
- Dataset YOLO: `../../../artifacts/datasets/station-photo-set-50-v1/`

Gunakan `scripts/prepare_station_training_set.py` untuk membangun ulang intake dan anotasi gate dari JPG/JSON LabelMe di folder ini. Foto tanpa JSON menghasilkan anotasi kosong, tetapi tidak otomatis memperoleh status review manusia.
