# Model-First Mobile-Ready Tactile Design

Tanggal: 8 September 2026

## Tujuan

Menghasilkan model segmentasi `tactile_paving` yang terukur pada data stasiun Indonesia, dapat diekspor ke LiteRT/TFLite, memiliki parity yang dapat diperiksa terhadap checkpoint PyTorch, dan dibungkus dengan kontrak runtime yang cukup jelas untuk integrasi aplikasi kamera Android. Status akhir pekerjaan ini adalah *mobile integration ready*, bukan sertifikasi keselamatan atau izin penggunaan berjalan mandiri.

Kebutuhan operasi untuk paket model dan decision layer berikutnya:

- memandu pengguna mengikuti koridor jalur tactile/pemandu tunanetra;
- mengumumkan arah kiri/kanan hanya ketika perubahan arah stabil dan cukup yakin;
- mendeteksi orang atau penghalang yang memotong koridor tactile;
- memberi status `STOP` untuk lubang/drop-off dan tepi peron dekat rel;
- memakai jarak 5 m sebagai target observasi ketika sensor kedalaman dan kondisi visual mendukung, bukan sebagai jaminan keselamatan;
- mendukung skenario demo terkontrol sampai ambang pintu gerbong Commuter, lalu meminta konfirmasi sebelum melewati celah/ambang.

Model satu kelas saat ini hanya mengenali tactile paving. Deteksi orang, koper/objek, lubang, tepi peron, pintu, dan celah kereta memerlukan kelas tambahan atau sensor/aturan terpisah; kebutuhan itu tidak boleh disamarkan sebagai kemampuan model tactile.

## Keputusan utama

- Model diselesaikan lebih dahulu sebelum source aplikasi kamera dibuat.
- Taxonomy tetap satu kelas: `tactile_paving`.
- Dataset test lokal yang sudah ada tetap menjadi holdout dan tidak dipakai untuk tuning.
- Checkpoint GuideTWSI hanya digunakan sebagai inisialisasi yang provenance dan lisensinya tercatat.
- Dataset eksternal hanya dapat masuk training setelah sumber, lisensi, checksum, taxonomy, dan aturan split lolos gate.
- Semua output sebelum gate final tetap berstatus `candidate`.
- Target mobile pertama adalah LiteRT/TFLite input 320; varian 416 hanya dipertahankan jika memberi peningkatan akurasi yang terukur dan masih memenuhi batas perangkat.
- Runtime aplikasi harus gagal tertutup: checksum salah, metadata tidak cocok, output stale, atau inference gagal menghasilkan status tidak tersedia/`Uncertain`, bukan arahan berjalan.

## Cakupan

### Termasuk

1. Audit prediksi kandidat v1 per gambar pada validation set.
2. Normalisasi gambar yang rusak tanpa mengubah identitas dataset diam-diam.
3. Intake dataset v2 dengan provenance, lisensi, deduplikasi, split, dan review label.
4. Training kandidat baru yang reproducible.
5. Evaluasi box dan mask, termasuk IoU per gambar dan failure cases.
6. Kalibrasi threshold hanya pada validation set.
7. Evaluasi final satu kali pada test holdout setelah konfigurasi dikunci.
8. Export model mobile FP32 dan INT8.
9. Parity check `.pt` terhadap LiteRT/TFLite pada input yang sama.
10. Paket integrasi Android berisi model, manifest, label, threshold, checksum, dan contoh kontrak tensor.

### Tidak termasuk

- Klaim bahwa model aman untuk navigasi mandiri.
- Integrasi UI kamera Flutter/Android pada fase model-first ini.
- Implementasi penuh ARCore Depth, object detector, TTS, haptic, atau decision engine pada fase model-first ini; kontrak dan gate untuk komponen tersebut tetap didefinisikan.
- Pemakaian foto test untuk pemilihan hyperparameter.
- Pelatihan dari gambar sintetis sebagai pengganti data test dunia nyata.

## Arsitektur pipeline

```text
approved sources
      |
      v
intake + provenance + license gate
      |
      v
decode/normalize -> content hash -> duplicate/leakage gate
      |
      v
LabelMe polygon -> human review -> YOLO segmentation export
      |
      v
train/validation tuning -> locked configuration -> test evaluation
      |
      v
candidate verifier -> FP32/INT8 export -> parity verifier
      |
      v
artifacts/mobile/<model-version>/
```

Satu manifest versi dataset menjadi sumber kebenaran untuk seluruh tahap. Setiap transformasi gambar menghasilkan file baru dan mencatat hash sumber serta hash hasil; anotasi ditransformasikan secara deterministik bila dimensi berubah. Duplikat konten dan turunan dari gambar yang sama tidak boleh melintasi split.

## Dataset v2

Dataset v2 menggabungkan data stasiun Indonesia yang telah direview dengan data tactile publik yang lolos gate lisensi. Data publik membantu representasi tekstur dan bentuk; data lokal tetap menentukan kemampuan domain stasiun Indonesia.

Setiap sampel wajib mencatat:

- sumber dan URL asal;
- lisensi atau izin penggunaan;
- checksum file sumber;
- lokasi/konteks stasiun;
- kondisi cahaya, sudut kamera, occlusion, dan tipe permukaan;
- status positif/negatif;
- split dan group id;
- checksum anotasi;
- reviewer dan waktu review.

Kelompok kondisi minimum untuk training dan validation:

- jalur lurus, belok, bercabang, dan warning block;
- jalur tertutup sebagian oleh orang, koper, atau objek;
- indoor, outdoor, siang, dan cahaya rendah;
- perspektif dekat dan jauh dari kamera setinggi tangan/dada;
- lantai kuning non-tactile dan pola ubin mirip tactile sebagai hard negative;
- peron, hall, gerbang, dan akses trotoar stasiun.
- orang, koper, troli, dan penghalang yang menutup sebagian atau seluruh koridor;
- percabangan dan perubahan arah untuk menguji perintah kiri/kanan;
- tepi peron, lubang, drop-off, celah, pintu, dan ambang gerbong sebagai kasus bahaya/endpoint terpisah.

Dataset tidak dianggap siap hanya karena mencapai jumlah tertentu. Gate memerlukan semua sampel valid, seluruh positif memiliki polygon, seluruh negatif telah dikonfirmasi, tidak ada leakage lintas split, dan distribusi kondisi dilaporkan. Target koleksi awal adalah sekurangnya 100 gambar positif lokal/publik yang dapat digunakan dan sekurangnya 50 hard negative; jumlah dapat bertambah berdasarkan failure analysis.

## Audit kandidat v1

Kandidat `tactile-one-class-v1` dipertahankan sebagai baseline tetap. Evaluator menjalankan inference pada validation set dan menyimpan untuk setiap gambar:

- mask ground truth dan prediksi;
- confidence;
- true positive, false positive, dan false negative;
- intersection, union, dan IoU;
- overlay visual;
- waktu inference;
- kategori failure case.

Audit tidak mengubah checkpoint. Hasilnya menentukan kebutuhan data v2, bukan menjadi alasan menurunkan gate.

## Training kandidat v2

Training menggunakan notebook/runner yang mem-pin versi dependency, seed, checkpoint awal, dataset version, image size, augmentasi, dan seluruh argumen Ultralytics. Training dilakukan dalam dua tahap jika dataset eksternal tersedia:

1. adaptasi satu kelas pada data tactile publik yang disetujui;
2. fine-tuning pada domain stasiun Indonesia.

Jika data eksternal tidak lolos gate lisensi, training langsung memakai data lokal v2 dan keputusan tersebut dicatat. Augmentasi dibatasi pada perubahan yang realistis: exposure, blur ringan, perspective kecil, shadow, dan occlusion. Transformasi yang menghapus tekstur tactile atau menghasilkan geometri fisik tidak masuk akal ditolak.

Pemilihan epoch, augmentasi, image size, dan threshold hanya memakai validation. Test holdout dibuka setelah satu konfigurasi dan satu checkpoint dikunci. Setiap eksperimen memiliki run id berbeda dan tidak boleh menimpa kandidat sebelumnya.

## Evaluasi dan gate model

Metrik yang disimpan:

- mask precision dan recall;
- mask mAP50 dan mAP50-95;
- IoU per gambar, mean IoU, dan median IoU;
- false positive pada hard negative;
- false negative pada jalur tertutup;
- latency CPU/GPU untuk perbandingan eksperimen;
- failure cases per kondisi.

Gate untuk melanjutkan ke export mobile:

1. bundle kandidat lolos checksum, task, label, dan versi dataset;
2. mean IoU local test sekurangnya 0,75;
3. mask recall local test sekurangnya 0,90;
4. tidak ada critical miss pada 20 pengulangan masing-masing skenario demo minimum yang didefinisikan di bawah;
5. tidak ada class-index mismatch;
6. seluruh failure case yang diketahui dicatat pada model card;
7. kandidat mengungguli baseline v1 pada validation dengan protokol yang sama.

Jika test set terlalu kecil untuk mendukung kesimpulan stabil, hasil dicatat sebagai tidak cukup bukti dan koleksi data dilanjutkan. Gate tidak diturunkan agar model dapat diekspor.

Skenario demo minimum untuk butir critical miss adalah: (a) koridor tactile lurus tanpa penghalang, (b) koridor sebagian tertutup orang/koper, (c) perubahan arah kiri atau kanan, dan (d) hard negative lantai kuning non-tactile. Critical miss berarti koridor tidak terdeteksi, koridor salah sehingga arahan berlawanan, atau bahaya yang terlihat tidak menghasilkan status `STOP`/`Uncertain`. Uji tepi peron dan ambang gerbong adalah uji sistem keselamatan terpisah, bukan bukti bahwa segmentasi tactile saja menjamin pengguna aman.

## Kontrak panduan dan keselamatan untuk tahap aplikasi

Decision layer yang mengonsumsi mask tactile harus menghitung koridor, deviasi arah, dan stabilitas lintas frame. Perintah `kiri`/`kanan` hanya boleh keluar setelah minimal 3 frame berturut-turut konsisten, confidence dan lebar koridor melewati threshold, serta tidak ada hazard yang lebih prioritas. Output yang hilang, kedaluwarsa, ambigu, atau berada di luar jangkauan menjadi `Uncertain`/`STOP`, bukan tebakan arah.

Jarak 5 m adalah target observasi. Dengan ARCore Depth, akurasi terbaik berada kira-kira pada 0,5–5 m dan bergantung pada gerakan, tekstur, serta dukungan perangkat; tanpa depth yang valid aplikasi tidak boleh mengucapkan jarak numerik. Peron dekat rel, lubang, drop-off, celah, dan ambang pintu selalu memiliki prioritas keselamatan di atas mengikuti tactile. Endpoint “masuk gerbong” hanya boleh dijalankan dalam demo terkontrol dengan konfirmasi pengguna dan pengawasan manusia.

## Export mobile

Setelah gate model lolos, exporter menghasilkan dua kandidat:

- LiteRT/TFLite FP32 untuk referensi parity;
- LiteRT/TFLite INT8 dengan representative calibration set yang hanya berasal dari train split.

Input awal 320x320. Varian 416x416 hanya dievaluasi bila 320 kehilangan jalur kecil secara material. Exporter memuat ulang setiap model dan menolak file kosong, task/label salah, tensor tidak dikenal, atau inference gagal.

## Parity check

Parity menggunakan gambar yang sama dan preprocessing yang sama untuk `.pt`, FP32, dan INT8. Laporan menyimpan:

- bentuk, dtype, dan quantization tensor input/output;
- jumlah mask setelah post-processing;
- confidence dan class id;
- IoU mask `.pt` terhadap hasil mobile;
- latency dan ukuran file;
- daftar sampel yang melampaui toleransi.

Gate parity awal:

- seluruh model dapat dimuat dan menjalankan inference;
- label tepat `{0: "tactile_paving"}`;
- tidak ada output NaN/Inf;
- median mask IoU FP32 terhadap `.pt` sekurangnya 0,95 pada sampel yang terdeteksi keduanya;
- median mask IoU INT8 terhadap `.pt` sekurangnya 0,90;
- penurunan recall INT8 terhadap `.pt` tidak lebih dari 0,03 pada validation;
- checksum dan metadata paket cocok.

Jika backend export yang dipilih tidak mempertahankan post-processing yang identik, parity dilakukan pada tensor mentah serta post-processor referensi yang sama dan perbedaan tersebut dicatat dalam manifest.

## Paket integrasi Android

Folder final `artifacts/mobile/<model-version>/` berisi:

```text
model_fp32.tflite
model_int8.tflite
labels.txt
model_manifest.json
thresholds.json
sha256.txt
MODEL_CARD.md
parity_report.json
evaluation_report.json
sample_input.jpg
sample_expected.json
```

`model_manifest.json` mendefinisikan:

- model id dan versi;
- status `mobile_integration_ready`;
- task dan class mapping;
- input width, height, color order, dtype, normalization, dan orientation policy;
- output tensor names/shapes dan post-processing version;
- confidence/mask thresholds;
- dataset version dan training run id;
- versi exporter/runtime;
- checksum seluruh file;
- batas kemampuan dan minimum Android API 24.

Paket tidak menyatakan `mobile_integration_ready` sebelum model gate dan parity gate sama-sama lolos.

## Error handling

- Gambar gagal decode: karantina sampel dan hentikan export dataset.
- Lisensi/provenance tidak lengkap: sampel tidak dapat masuk training.
- Duplikat lintas split: dataset ditolak.
- Label kosong pada sampel positif atau polygon di luar batas: dataset ditolak.
- Training terputus: resume hanya dari checkpoint dengan run/config yang cocok.
- Candidate directory sudah ada: gunakan run id baru; jangan overwrite.
- Export gagal atau model tidak dapat dimuat ulang: paket mobile tidak dibuat.
- Parity di bawah toleransi: model mobile ditolak dan `.pt` tetap kandidat riset.
- Checksum runtime tidak cocok: aplikasi harus menolak model.

## Pengujian

- Unit test untuk normalisasi gambar, transformasi polygon, manifest, deduplikasi, dan leakage gate.
- Unit test evaluator IoU dengan mask sintetis deterministik.
- Contract test notebook/runner training.
- Candidate bundle verifier pada bundle valid dan setiap jenis kerusakan metadata.
- Export smoke test yang memuat ulang FP32 dan INT8.
- Parity regression test pada sample fixture yang dipaketkan.
- Manifest contract test untuk konsumen Android.
- Full test suite lokal sebelum setiap commit kandidat atau paket mobile.

## Bukti selesai

Fase model-first selesai hanya bila:

1. dataset v2 tervalidasi dan dapat direproduksi;
2. kandidat final memenuhi model gate pada test holdout;
3. model FP32 dan INT8 memenuhi parity gate;
4. paket integrasi Android lengkap dan lolos verifier;
5. model card menyatakan batas kemampuan dan residual risk;
6. commit lokal dan GitHub identik.

Source aplikasi kamera menjadi subproyek berikutnya dan hanya mengonsumsi paket yang berstatus `mobile_integration_ready`.
