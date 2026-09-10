# Handoff Model Tactile V3

Dokumen ini adalah sumber konteks utama untuk teman atau Codex yang melanjutkan pekerjaan. Baca seluruh dokumen sebelum mengubah data, training, evaluasi, atau integrasi mobile.

## Tujuan produk

Target akhirnya adalah aplikasi kamera mobile yang membantu pengguna tunanetra:

- tetap mengikuti jalur tactile/blind pedestrian path;
- mendapat arahan `Lurus`, `Belok kiri`, atau `Belok kanan` setelah arah stabil;
- mengetahui orang atau penghalang yang memotong koridor berjalan;
- menerima perintah `STOP` saat ada lubang, drop-off, tepi peron, rel, atau celah kereta;
- menggunakan target observasi hingga **5 meter** jika ada depth dan kalibrasi yang memadai;
- menyelesaikan demo terkontrol hingga masuk gerbong Commuter.

Lima meter adalah target observasi, bukan jaminan keselamatan dari kamera monokular. Melintasi celah peron–kereta harus membutuhkan konfirmasi pengguna atau pengawasan manusia. Hazard selalu mengalahkan instruksi arah. Data hilang, frame basi, confidence rendah, atau keadaan ambigu harus menghasilkan `STOP` atau `Uncertain`.

## Batas model yang sedang dikerjakan

Model saat ini adalah instance segmentation satu kelas: `tactile_paving`. Model ini hanya ditargetkan menghasilkan mask koridor tactile yang stabil. Model ini belum mendeteksi orang, penghalang, lubang, tepi peron, rel, pintu kereta, celah kereta, arah belok, atau jarak.

Fungsi-fungsi tersebut memerlukan model hazard/obstacle terpisah, depth atau distance estimation, temporal stability, dan decision layer. Karena itu kandidat saat ini berstatus **not mobile-ready** sampai seluruh gate terkait lulus.

## Progres dan score sementara

Kandidat selesai: `tactile-one-class-v3-public4-station1`.

- Training: 80 epoch, sekitar 28 menit, Google Colab Tesla T4.
- Inisialisasi: `models/guidetwsi/yolo11n_tactile.pt`.
- Dataset: campuran data publik dan station dengan rasio 4:1.
- Confidence protected test yang terkunci: `0.05`.

### Training validation

| Metric | Score |
| --- | ---: |
| Box precision | 90.05% |
| Box recall | 73.54% |
| Box mAP50 | 83.78% |
| Box mAP50-95 | 69.84% |
| Mask precision | 89.58% |
| Mask recall | 73.85% |
| Mask mAP50 | **82.50%** |
| Mask mAP50-95 | **69.79%** |

### Protected test

Protected test saat ini berisi 17 gambar: 11 positif dan 6 negatif. Angka ini hanya hasil sementara dan belum cukup untuk klaim keselamatan.

| Metric | Score sementara | Target | Status |
| --- | ---: | ---: | --- |
| Mask recall | **100.00%** | >= 90% | Lulus |
| Positive mean IoU | **56.41%** | >= 75% | Gagal |
| Precision | **64.71%** | Informasi | Tercatat |
| False-positive rate | **100.00%** | <= 50% | Gagal |
| Median CPU latency | 92.1 ms | Belum ditetapkan | Sementara |

Hasil keseluruhan: **rejected / not mobile-ready**. Tidak ada satu angka “akurasi” yang sah untuk merangkum semuanya. mAP, recall, kualitas overlap, dan false-positive rate mengukur perilaku berbeda. Mask recall memang mencapai 100%, tetapi semua 6 gambar negatif masih menghasilkan false positive dan mean IoU belum mencapai target.

## Training yang belum selesai

Target berikutnya adalah `tactile-one-class-v3-public2-station1` dengan rasio publik:station 2:1. Training lokal sebelumnya dihentikan saat epoch 5 dan bukan kandidat selesai. Jangan memakai atau menyebut checkpoint parsial tersebut sebagai hasil akhir. Kandidat 2:1 harus dilatih ulang dari checkpoint GuideTWSI menggunakan notebook Colab yang disediakan.

## Branch dan notebook

- Repository: `Riqqi15/Yoloooo`
- Branch: `codex/model-first-mobile-ready`
- Notebook: `notebooks/train_tactile_v3_public_colab.ipynb`
- Ultralytics: `8.4.138`
- Seed: `42`
- Image size: `640`
- Epoch maksimum: `80`
- Batch: `8`
- Device wajib: CUDA GPU

Notebook harus tetap bersih, tanpa output tersimpan, tanpa cell percobaan, tanpa upload manual, dan tanpa akses ke protected ground truth. Dari runtime Colab baru, pilih GPU lalu jalankan semua cell secara berurutan.

## Inventaris data dan artefak

| Path | Isi |
| --- | --- |
| `models/guidetwsi/yolo11n_tactile.pt` | Checkpoint awal GuideTWSI |
| `data/public/guidetwsi-rbar-v1/guidetwsi-rbar-v1.zip` | Cache publik immutable, 3.960 file, 345.332.596 byte |
| `data/public/guidetwsi-rbar-v1/sha256.txt` | SHA-256 ZIP publik |
| `data/public/guidetwsi-rbar-v1/provenance.json` | Provenance sumber publik |
| `data/training/manifests/guidetwsi-rbar-2k-v1.json` | Manifest seleksi publik |
| `artifacts/datasets/station-tactile-v2` | Dataset station yang sudah direview |
| `artifacts/candidates/tactile-one-class-v3-public4-station1` | Kandidat 4:1 selesai dan ditolak gate |

ZIP publik memiliki SHA-256 `4796d2eea993f62b2db0934b3fe639c4d33b7bf481281509d33e3901c8dbd99e` dan tercatat berlisensi CC0/Public Domain. File besar disimpan melalui Git LFS. Mengganti ZIP mewajibkan pembaruan checksum, provenance, manifest, dan jumlah file secara bersamaan.

Foto sumber yang memiliki trolley sengaja dikeluarkan berdasarkan review pengguna. Jangan memasukkan kembali sumber trolley tersebut secara diam-diam. Trolley dan penghalang lain nantinya harus masuk ke dataset obstacle yang terpisah dan direview.

## Yang harus dilakukan penerus

1. Buka notebook dari branch `codex/model-first-mobile-ready` di Google Colab.
2. Pilih runtime GPU dan pastikan `torch.cuda.is_available()` bernilai benar.
3. Pilih **Jalankan semua**. Notebook akan mengambil LFS yang diperlukan, memeriksa checksum ZIP, mengekstrak 3.960 file, membangun dataset 2:1, lalu training.
4. Jangan mengubah rasio, seed, split, threshold, atau hyperparameter dalam eksperimen pembanding pertama.
5. Setelah selesai, pastikan bundle berisi `best.pt`, `metrics.json`, `training_config.json`, `MODEL_CARD.md`, dan `sha256.txt`.
6. Unduh `tactile-one-class-v3-public2-station1.zip` dari cell terakhir.
7. Kirim ZIP kandidat kepada pemilik repository. Jangan mengunggah protected ground truth ke Colab atau GitHub.
8. Pemilik repository menjalankan protected evaluation lokal satu kali menggunakan threshold yang dipilih dari validation biasa.
9. Bandingkan kandidat 2:1, kandidat 4:1, dan baseline memakai seluruh gate di bawah.

## Gate sebelum mobile export

Semua syarat berikut wajib lulus bersamaan:

- candidate bundle lengkap dan checksum valid;
- class map hanya `tactile_paving` dan sesuai pipeline;
- protected/local-test mask recall >= 0.90;
- protected/local-test positive mean IoU >= 0.75;
- protected/local-test false-positive rate <= baseline 0.50;
- kandidat mengalahkan baseline tanpa class mismatch;
- failure case dicatat di model card;
- nol critical miss dalam 20 pengulangan untuk jalur lurus, jalur tertutup orang/tas, belok kiri/kanan, dan yellow non-tactile hard negative.

Training mAP tinggi, file export berhasil dibuat, atau satu demo berhasil bukan bukti bahwa gate telah lulus.

## Roadmap improvement

### P0 — selesaikan perbandingan model

- Latih kandidat 2:1 dengan konfigurasi terkunci dan validasi bundle.
- Catat commit, versi dataset, checksum manifest/checkpoint, seed, GPU, durasi, epoch, dan semua metric.
- Pilih threshold hanya dari validation biasa, bekukan, lalu jalankan protected test satu kali.
- Pilih kandidat berdasarkan semua gate, bukan hanya training mAP.
- Simpan `last.pt` atau salinan Drive opsional agar interupsi Colab dapat dilanjutkan; checkpoint parsial tetap berstatus incomplete.

### P0 — perbaiki false positive dan mask

- Prioritaskan hard negative legal: ubin kuning biasa, garis cat, marka peron, sambungan lantai, bayangan, rel, tepi peron, dan tekstur berulang mirip tactile.
- Audit polygon untuk jalur tipis/jauh, belokan, persimpangan, perspective narrowing, partial visibility, occlusion, low light, blur, glare, dan cropped path.
- Jangan menyalin gambar atau label protected test ke training. Cari contoh training baru yang secara visual serupa.
- Gunakan augmentasi yang realistis dan tidak merusak geometri tactile.

### P1 — perkuat data dan evaluasi

- Perluas validation station; manifest 4:1 sekarang hanya memiliki 13 contoh station validation.
- Pisahkan data berdasarkan station, source group, atau capture sequence dan audit hash/near-duplicate agar scene terkait tidak bocor antar-split.
- Beri tag skenario dan laporkan metric per kondisi: lurus, belok, persimpangan, jauh, occlusion, gelap, blur, glare, permukaan kuning non-tactile, tepi peron, dan ramai.
- Perluas protected set dari 17 gambar sambil menjaganya immutable dan tidak dapat diakses notebook training.
- Ulangi konfigurasi pemenang dengan seed tambahan ketika kuota GPU tersedia.
- Simpan overlay kegagalan dan bedakan model error, label error, serta scene ambigu.
- Catat image-level false-positive rate dan jumlah region false positive.

### P1 — validasi navigasi video

- Buat walkable corridor dari mask hanya setelah gate segmentasi lulus.
- Butuhkan minimal tiga frame confident berturut-turut sebelum mengumumkan belok.
- Catat direction accuracy, path-departure rate, hazard recall, false alarm per menit, warning lead time, stale-frame detection, end-to-end latency, dan recovery setelah occlusion.
- Uji 20 kali per skenario; nilai rata-rata tidak boleh menyembunyikan critical failure.
- Uji target 5 meter dengan jarak terukur, tinggi/sudut kamera berbeda, siang/malam, motion blur, dan beberapa HP.

### P2 — tambahkan komponen keselamatan

- Tambahkan detector orang/obstacle terpisah dan cek apakah deteksinya memotong koridor tactile.
- Buat kelas/model terpisah untuk lubang/drop-off, tepi peron/rel, pintu kereta, dan celah peron–kereta.
- Tambahkan depth terkalibrasi sebelum memberikan peringatan berbasis meter.
- Sediakan state audio/haptic yang berbeda: `Straight`, `Left`, `Right`, `Obstacle`, `STOP`, dan `Uncertain`.
- Benchmark langsung di HP: inference, camera pipeline, thermal throttling, memori, baterai, audio/haptic delay, serta kesetaraan FP32/FP16/INT8.

### P2 — validasi pengguna dan governance

- Rancang instruksi bersama pengguna blind/low-vision; uji bertahap dari video, mock route, closed route, protected platform simulation, lalu station trial berizin dan diawasi.
- Jangan memulai live test di dekat rel aktif. Gunakan sighted safety observer, exclusion zone, prosedur berhenti, dan manual override.
- Uji hujan, gelap, refleksi, crowd, kamera tertutup, frame drop, overheating, low battery, loss of depth, dan out-of-distribution scene. Semua degradasi harus fail closed.
- Lindungi wajah, suara, dan metadata lokasi; dapatkan informed consent untuk studi pengguna.
- Versioning wajib mencakup dataset, manifest, model, threshold, calibration profile, dan decision rules.
- Pertahankan provenance/lisensi, automated integrity checks, failure log, regression set, rollback, dan kill switch.

## Batas keselamatan

Ini adalah riset/prototipe offline. Jangan memakai kandidat sekarang sebagai alat navigasi mandiri atau sebagai bukti bahwa berjalan di peron/rel aman. Mobile integration tetap ditunda sampai segmentation gate lulus; hazard model dan supervised system testing tetap wajib sesudahnya.
