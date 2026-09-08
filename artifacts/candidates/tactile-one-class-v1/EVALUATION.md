# Evaluasi baseline tactile-one-class-v1

## Status

- Integritas bundle: valid.
- Task: segmentasi satu kelas `tactile_paving`.
- Dataset: `station-photo-set-50-v1`.
- Keputusan: baseline riset saja; tidak layak untuk deployment atau panduan berjalan.

## Konfigurasi

- Ultralytics: 8.4.138
- Epoch: 80
- Ukuran input: 640
- Seed: 42
- Checkpoint SHA-256: `7ec97dc07ad3527247122dbe549fd0cae5ac3c82e111becaa18855efe6511bdf`

## Metrik validation

| Metrik | Box | Mask |
|---|---:|---:|
| Precision | 0.0031 | 0.0031 |
| Recall | 0.2500 | 0.2500 |
| mAP50 | 0.0188 | 0.0188 |
| mAP50-95 | 0.0132 | 0.0094 |

Fitness: 0.0226.

## Interpretasi

Model hanya menemukan sebagian kecil instance jalur taktil dan menghasilkan terlalu banyak prediksi salah. Validation set juga sangat kecil, sehingga angkanya belum stabil. Menambah epoch saja tidak cukup untuk memperbaiki masalah ini.

Dataset saat ini berisi 50 gambar, 30 polygon taktil, dan split 35 train / 8 validation / 7 test. Hanya 13 gambar yang positif taktil, sehingga keragaman contoh positif masih menjadi batas utama.

## Langkah berikutnya

1. Simpan kandidat ini sebagai baseline pembanding.
2. Inspeksi visual false positive dan false negative pada validation set.
3. Normalisasi ulang tiga JPEG validation yang diperbaiki otomatis oleh Ultralytics saat scan.
4. Tambah foto positif yang beragam sebelum tuning berikutnya.
5. Jangan memakai test split untuk memilih epoch, augmentasi, atau threshold.
