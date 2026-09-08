# Audit Mismatch dan Roadmap AI Camera

Tanggal audit: 5 September 2026; diperbarui 7 September 2026. Cakupan pemeriksaan: workspace `C:\Users\riyadh\Downloads\AI Camera`, kode Python, dokumentasi proyek, manifest, dan laporan evaluasi yang tersedia. Repo aplikasi Flutter/Android lain tidak diperiksa.

Dokumen ini menjadi acuan perbaikan berikutnya. Checklist kosong berarti pekerjaan belum selesai; pembuatan dokumen ini tidak mengimplementasikan perbaikannya. Plan lama di `docs/superpowers/plans` mencatat milestone terbatas dan tidak membuktikan seluruh fitur dalam MD utama selesai.

## 1. Kesimpulan dan scope yang disepakati

Kode sesuai sebagai baseline eksperimen lokal. Model dan runtime belum memenuhi seluruh kebutuhan pemandu kamera dalam rancangan awal.

- Termasuk: bangunan stasiun, hall, gerbang, ruang tunggu, peron, area rel untuk analisis gambar, pintu masuk/keluar, serta trotoar umum yang menjadi akses atau terhubung langsung dengan stasiun.
- Dikecualikan dari dataset aktif: foto bus/halte sebagai konteks utama, jalan umum, dan trotoar tanpa hubungan dengan stasiun.
- Kendaraan kebetulan terlihat di sekitar stasiun tidak otomatis membatalkan konteks stasiun. Scope dinilai dari konteks adegan, bukan nama kelas hasil deteksi.
- Scope lingkungan tidak sama dengan taxonomy model. Model COCO mengenali bus bukan bukti bahwa dataset proyek berisi bus atau sudah digunakan untuk training.
- Semua 17 foto aktif tetap `test_only`. Riwayat inspeksi/evaluasinya sudah ada; jangan gunakan sebagai data training atau untuk memilih hyperparameter. Untuk estimasi generalisasi final, tambah holdout baru yang belum dipakai selama pengembangan.
- Inferensi saat ini CPU laptop. Benchmark mobile, ARCore, dan integrasi Flutter tetap tahap terpisah; tidak dilakukan dengan mengubah repo aplikasi lain pada sesi ini.

## 2. Bukti progres yang tersedia

| Area | Bukti lokal | Status |
|---|---|---|
| Object baseline | `scripts/baseline_object_inference.py`, `models/yolo11n.pt` | Image/stream inference, koridor trapesium, overlay, STOP, timing tersedia |
| Tactile baseline | `scripts/baseline_tactile_inference.py`, `models/guidetwsi/yolo11n_tactile.pt` | Segmentasi satu kelas, mask, pengecekan task/label, timing tersedia |
| Batch dan scope | `scripts/evaluate_samples.py`, `data/dataset_scope.json` | Daftar kurasi 17 foto; trotoar akses masuk; file tak terdaftar ditolak sebelum model dimuat |
| Anotasi | `scripts/import_labelme_ground_truth.py`, `scripts/ground_truth_manifest.py` | 17 anotasi memiliki provenance/hash dan telah direview Riyadh; 11 positif, 6 negatif |
| Evaluasi | `scripts/evaluate_ground_truth.py`, `runs/accuracy/tactile_metrics.json` | Evaluasi resmi aktif; accuracy 70,59%, precision 75%, recall 81,82%, F1 78,26%, mean IoU 0,5318 |
| Kontrak model | `models/baseline_manifest.json`, `artifacts/BASELINE_MODEL_CARD.md` | Path/size/hash/task/raw labels diverifikasi sebelum load; batas taxonomy terdokumentasi |
| Tes | `tests/` | Run 7 September 2026: 57 tes lulus. Ini tes fitur yang sudah ditulis, bukan bukti semua kebutuhan sudah tercakup |
| Training/export | `notebooks/`, `artifacts/` | Belum ada notebook training atau artefak final mobile |
| Runtime lengkap | `implementation/*.md` | Decision engine, watchdog, fusion, feedback, capability profile, dan integrasi Android masih rancangan |

### Angka evaluasi historis yang tidak boleh dianggap aktif

Laporan lama yang kini berada di `runs/accuracy/archive/` menggunakan 10 foto: 8 berlabel positif dan 2 negatif; TP 6, FP 1, FN 2, TN 1. Accuracy keberadaan 70%, precision keberadaan 85,71%, recall keberadaan 75%, F1 keberadaan 80%; mean IoU pada 8 gambar GT positif sekitar 0,09697.

Angka precision/recall di atas bukan precision/recall pixel mask atau evaluasi objek per kelas. Mean IoU tersebut bukan otomatis definisi mIoU dataset pada seluruh kelas. Semua label masih AI-authored/provisional. Label salah atau polygon meleset dapat menurunkan IoU; belum tepat menyimpulkan bahwa model adalah satu-satunya penyebab nilai rendah. Validasi ukuran dan nilai 0/255 tidak memvalidasi lokasi polygon.

## 3. Daftar mismatch dan dampaknya

P0: harus dibereskan sebelum angka evaluasi dipercaya. P1: sebelum training/release kandidat model. P2: sebelum integrasi/demo.

| ID | Prioritas | Mismatch yang teramati | Perbaikan yang diperlukan |
|---|---|---|---|
| M01 | P0 | Importer otomatis mengisi `reviewed`; manifest berisi catatan AI provisional, tetapi evaluator hanya memeriksa status/format | Pisahkan asal anotasi, kelengkapan format, dan persetujuan reviewer manusia. Default evaluator resmi menolak anotasi provisional |
| M02 | P0 | Polygon awal belum memiliki bukti QA manusia terhadap gambar asli | Audit seluruh 17 foto, koreksi koordinat/occlusion, dan tandai ambigu sebagai belum diputuskan; jangan mengubah ketidakpastian menjadi negatif |
| M03 | P0 | Checkpoint menyimpan `{0: braille}`, konfigurasi upstream yang dicatat kode `{0: tactile_paving}`, sementara MD menargetkan dua kelas | Verifikasi kontrak checkpoint pada sumber/version yang tepat; pertahankan raw label. Alias hanya dengan bukti; dua kelas membutuhkan anotasi dan checkpoint baru |
| M04 | P0 | Evaluator belum menolak secara eksplisit setiap row inferensi gagal; metrik keberadaan memakai default hitungan nol, dan mask prediksi hanya dibaca untuk GT positif | Tolak row gagal/incomplete, pastikan cakupan sumber tepat, baca semua mask prediksi, dan cocokkan laporan dengan versi input/model |
| M05 | P1 | SHA-256 tactile dihitung untuk laporan, belum dibandingkan terhadap nilai tepercaya sebelum load; object model belum memiliki kontrak artefak setara | Manifest model berversi berisi hash, task, raw labels, sumber/revisi, dan batas kemampuan. Tolak mismatch sebelum inference |
| M06 | P1 | Gate scope berada pada batch dan berdasarkan nama file; isi file dengan nama sama dapat berganti. Belum ada pipeline split/deduplikasi | Tambah identitas konten dan metadata sesi/lokasi pada dataset; terapkan validasi di jalur dataset training/evaluasi baru. CLI baseline tetap diberi label alat diagnostik |
| M07 | P1 | YOLO11n COCO belum memenuhi taxonomy 18 objek/bahaya dalam MD | Audit kelas yang didukung, siapkan label/data stasiun untuk kelas prioritas, dan ukur recall kelas kritis. Nama kelas baru saja tidak membuat model mampu mendeteksinya |
| M08 | P1 | Dokumen menampilkan contoh training/export, tetapi notebook, checkpoint fine-tune, dan hasil export belum tersedia | Implementasikan training yang reproducible, evaluasi, export, dan parity check secara berurutan setelah data lolos QA |
| M09 | P2 | STOP sederhana dan koridor statis belum setara state machine, fusion, stale rejection, quality gate, dan watchdog | Bangun decision engine deterministik dan replay test; lanjutkan watchdog serta integrasi feedback di runtime target |
| M10 | P2 | Timing CPU/inference dan CSV batch belum setara latency kamera-ke-feedback pada Android | Ukur pipeline lengkap pada HP, profil otomatis, backpressure, thermal, RAM, dan fallback |
| M11 | P1 | MD lama masih memuat konteks awal, angka 11 foto, dan asumsi kode Flutter yang tidak ada di workspace ini | Sinkronkan dokumen aktif; beri label historis pada laporan/plan lama dan bedakan scope dataset dari tempat demo terkontrol |
| M12 | P1 | Atomic replace per mask/CSV bukan transaksi satu batch: kegagalan validasi/penulisan setelah beberapa mask dapat meninggalkan mask berubah dengan manifest lama | Validasi awal, staging satu batch, dan mekanisme commit/rollback; uji kegagalan di tengah proses sebelum mengklaim seluruh batch atomik |

M03: `validate_model_contract` saat ini menerima raw label `braille`; perbedaannya dengan konfigurasi upstream hanya dicatat. Ini bukan pembuktian bahwa guiding path dan warning block sudah dibedakan.

M04 juga membutuhkan pelaporan `evaluation_status`, provenance label, versi dataset, hash input/mask, hash checkpoint, dan settings inference agar metrik tersimpan tidak terlihat sah setelah input berubah.

## 4. Urutan pekerjaan dan kriteria selesai

### Tahap A — Pulihkan keandalan evaluasi (M01, M02, M04)

File utama: `scripts/ground_truth_manifest.py`, `scripts/import_labelme_ground_truth.py`, `scripts/evaluate_ground_truth.py`, `data/ground_truth/README.md`, dan tes terkait.

- [x] Tambahkan provenance anotasi dan status review terpisah; migrasikan label AI lama sebagai provisional sambil menyimpan backup dan riwayat.
- [x] Import JSON tidak otomatis berarti persetujuan reviewer manusia. Identitas reviewer/waktu hanya dicatat dari tindakan review nyata, tidak dibuat-buat.
- [x] Default evaluasi resmi menolak provisional/ambiguous. Jika mode diagnostik provisional disediakan, gunakan output berbeda dan status eksplisit yang terbawa ke laporan.
- [x] Sediakan overlay QA yang menampilkan polygon terhadap gambar asli; audit semua 17 foto dalam koordinat sumber. Jangan menyalin prediction mask sebagai GT.
- [x] Reviewer manusia Riyadh memeriksa seluruh overlay dan memberi keputusan akhir, termasuk bahwa saluran air pada foto Sumberpucung bukan tactile.
- [x] Gambar ambigu tetap pending; negative hanya bila reviewer dapat menetapkan tidak ada tactile yang terlihat. Label hanya pixel terlihat, bukan area di balik orang/benda atau watermark.
- [x] Tolak laporan dengan inferensi gagal, data wajib hilang, sumber duplikat/tidak cocok, mask invalid, atau identitas input yang berubah.
- [x] Jalankan evaluasi ulang setelah review, simpan hasil resmi terpisah dari laporan provisional, dan catat perubahan label beserta alasannya.

**Selesai bila:** tes membuktikan label AI tidak bisa lolos sebagai review manusia hanya karena import berhasil; kegagalan inferensi tidak dihitung sebagai negative; setiap angka hasil resmi dapat ditelusuri ke label yang disetujui dan input yang cocok. Penyelesaian QA manusia memerlukan reviewer nyata; agen dapat menyiapkan alat dan draft koreksi.

### Tahap B — Tetapkan kontrak model dan rapikan dokumentasi (M03, M05, M11)

File utama: kedua script baseline, `README.md`, `NEW_CHAT_CONTEXT.md`, `training/*.md`, dan `implementation/*.md`. Manifest model dibuat sebagai bagian implementasi tahap ini.

- [x] Verifikasi sumber/revisi/lisensi dan metadata checkpoint terhadap sumber resmi terpin.
- [x] Catat raw label, arti label yang terbukti, task, hash, input, versi runtime, serta kelas yang belum didukung.
- [x] Bandingkan hash artefak dengan manifest tepercaya sebelum load; mismatch path, ukuran, atau hash ditolak.
- [x] Dokumentasikan baseline satu kelas tactile secara eksplisit. Target dua kelas tetap pekerjaan fine-tuning; metadata tidak diganti untuk mengesankan kemampuan yang belum ada.
- [x] Pisahkan kelas objek COCO yang tersedia dari kelas bahaya khusus stasiun yang masih membutuhkan data/training.
- [x] Sinkronkan scope stasiun dan trotoar akses pada seluruh MD aktif. Referensi ruang meeting hanya untuk latihan fungsi terkontrol, bukan domain dataset tambahan.

**Selesai bila:** checkpoint/label/hash salah ditolak dalam tes, model card baseline mencatat batas nyata, dan dokumen aktif tidak menyatakan fitur yang belum ada sebagai sudah selesai.

### Tahap C — Dataset training stasiun yang dapat diaudit (M06, M07, M12)

Acuan: `training/01-dataset-and-labeling.md`, `data/dataset_scope.json`, dan importer yang ada. Gunakan helper yang sudah tersedia; jangan memaksa manifest test-only menjadi manifest training.

- [x] Buat jalur intake data training baru beserta lokasi, sesi, konteks stasiun, sumber/izin penggunaan, dan hash konten.
- [ ] Tetapkan taxonomy objek prioritas dan tactile sebelum annotator bekerja; jangan menambah kelas yang tidak punya data/evaluasi memadai.
- [x] Saring duplikat persis dan kandidat near-duplicate; near-duplicate lintas split ditolak.
- [x] Split berdasarkan sesi/lokasi, simpan seed dan versi; tolak konten 17 foto test aktif masuk train/validation.
- [ ] Cakup peron, hall, trotoar akses, cahaya buruk, occlusion, serta hard negatives seperti cat kuning tanpa tekstur tactile.
- [x] Validasi polygon tactile pada gambar sumber dan uji konsistensi saat commit impor gagal.
- [x] Validasi bounding box/polygon dan ekspor YOLO atomik pada jalur intake training baru; export hanya menerima review manusia yang terikat hash.

**Selesai bila:** train/validation/test beserta distribusi kelas dapat diperiksa, provenance lengkap, tidak ada duplikat konten lintas split yang diketahui, dan dataset ekspor tervalidasi tanpa memakai foto test sebagai training.

### Tahap D — Training dan evaluasi kandidat (M07, M08)

Acuan: `training/02-colab-training-and-export.md`, `training/03-evaluation-and-model-safety.md`, folder `notebooks` dan `artifacts`.

- [ ] Buat notebook Colab dengan dependency terpin, dataset version, seed, command, log, dan checkpoint tersimpan.
- [ ] Latih detector objek stasiun serta fine-tune segmenter sesuai taxonomy yang telah disetujui pada Tahap C.
- [ ] Kalibrasi threshold memakai validation set; gunakan test holdout untuk evaluasi final.
- [ ] Ukur precision/recall per kelas, mAP50/50-95, confusion matrix, recall bahaya kritis, mask metrics, dan failure cases per kondisi.
- [ ] Laporkan definisi IoU secara eksplisit; jangan menyamakan metrik keberadaan gambar dengan kualitas pixel mask.

**Selesai bila:** training dapat direproduksi, baseline vs kandidat dibandingkan pada data valid yang sama, dan model card menjelaskan kemampuan/laporan kegagalan. Target riset dalam MD menjadi gate terukur, bukan klaim otomatis setelah training selesai.

### Tahap E — Export dan runtime yang dapat diuji (M08, M09)

- [ ] Export model kandidat ke target mobile, lakukan kalibrasi INT8 menggunakan data yang sesuai, lalu muat ulang dan bandingkan box/class/mask dengan `.pt`.
- [ ] Simpan manifest final: hash, label, task, input/output, threshold, versi dataset, metrik, dan batas kemampuan.
- [ ] Buat decision engine deterministik dengan timestamp/TTL, hazard priority, state transition, dan tactile stability; uji input sintetis serta replay timeline.
- [ ] Tambahkan quality gate dan watchdog independen: kamera freeze/gelap, hasil stale, model gagal, dan latency berlebihan harus diuji secara eksplisit.

**Selesai bila:** export parity berada pada toleransi yang ditetapkan sebelum pengujian, failure injection menghasilkan state yang benar, dan replay tidak memakai hasil stale. Core offline dapat disiapkan di workspace ini; integrasi repo aplikasi mengikuti cakupan sesi terpisah.

### Tahap F — Integrasi Android dan demo terkontrol (M09, M10)

- [ ] Integrasikan kedua model, lifecycle/backpressure, feedback priority, TTS cancellation, dan STOP audio/haptic.
- [ ] Tambahkan depth opsional dengan pengecekan capability serta fallback sesuai bukti sensor.
- [ ] Benchmark kamera-ke-feedback pada HP fisik: p95, dropped frames, RAM, thermal, battery, dan profil Basic/Balanced/Full/Degraded.
- [ ] Jalankan skenario serta pengulangan demo pada `implementation/04-demo-testing-and-observability.md`; perbarui risk register dan gate hasil.

**Selesai bila:** seluruh kebutuhan runtime yang diaktifkan memiliki bukti pada perangkat target dan skenario terkontrol. Pengujian ini tidak berarti izin penggunaan mandiri dekat rel/peron nyata.

## 5. Aturan pelaporan progres

- `Implemented`: kode/artefak ada; sebutkan file dan cakupannya.
- `Verified`: pemeriksaan relevan sudah dijalankan; sebutkan bukti dan tanggal/run.
- `Provisional`: data/hasil belum disetujui atau ada ketidakpastian semantik.
- `Pending`: pekerjaan belum dikerjakan.
- Jangan menghitung persentase seluruh proyek berdasarkan jumlah MD, checkbox, atau tes lulus.
- Persetujuan manusia, lisensi, training cloud, dan pengujian HP tidak dianggap selesai hanya karena script/helper dibuat.
- Saat konteks berubah, perbarui roadmap ini serta `NEW_CHAT_CONTEXT.md`; plan lama tetap sebagai catatan historis.

## 6. Fokus eksekusi selanjutnya

Tahap A dan B selesai untuk baseline lokal. Fokus berikut adalah Tahap C: intake dataset training stasiun terpisah, provenance lokasi/sesi/lisensi, deduplikasi, dan split anti-leakage. Jangan memakai 17 foto `test_only` sebagai train/validation.

Gate evaluasi telah diterapkan dan dilalui: label provisional tidak dapat dianggap review manusia, sedangkan 17 keputusan reviewer Riyadh menghasilkan laporan resmi yang dapat ditelusuri. Hasil ini tetap evaluasi offline kecil, bukan bukti keselamatan penggunaan.

## 7. Catatan eksekusi Tahap A

Eksekusi 5 September 2026:

- Manifest dan metrik lama diarsipkan di `data/ground_truth/history/` dan `runs/accuracy/archive/`.
- Schema manifest kini mencatat hash sumber/mask, asal anotasi, ID anotasi, status review, reviewer, dan waktu review.
- Import Labelme menghasilkan batch mask versioned dan status `provisional`; tes failure injection membuktikan kegagalan commit tidak mengubah manifest aktif atau meninggalkan batch final.
- Batch prediction mencatat hash sumber, scope, kedua model, dan prediction mask. Evaluator mencocokkan seluruh set sumber serta menolak row gagal, artefak hilang/berubah, dan mask invalid.
- Evaluator resmi menolak 10 anotasi AI. Mode diagnostik menghasilkan file provisional terpisah hanya ketika tidak ada row ambiguous.
- Overlay QA untuk 10 foto tersedia di `runs/ground-truth-review/`. Audit AI mengoreksi draft polygon pada `livery`, `stasiun-jaksel`, `Tanah Abang`, dan batas watermark gambar kecil. `images (4).jpg` ditandai ambiguous karena gelap dan tertutup berat.
- Status pada 5 September: 9 provisional, 1 ambiguous, 0 human-reviewed. Saat itu belum ada metrik yang dapat dianggap resmi.

Pada 5 September, Tahap A belum penuh karena review manusia dan evaluasi ulang belum terjadi. Kondisi ini diselesaikan pada eksekusi 7 September di bawah.

Eksekusi 7 September 2026:

- Dataset aktif bertambah menjadi 17 foto `test_only`; 17/17 inferensi berhasil.
- Riyadh menyelesaikan review seluruh anotasi: 11 positif, 6 negatif, tanpa provisional/ambiguous/incomplete.
- Foto Sumberpucung diputuskan negatif karena objek kuning yang tampak adalah saluran pembuangan air, bukan tactile paving.
- Manifest berstatus `official_ready: true`; evaluator resmi menulis `runs/accuracy/tactile_metrics.json`.
- Hasil resmi: TP 9, FP 3, FN 2, TN 3; accuracy 70,59%, precision 75%, recall 81,82%, F1 78,26%, mean IoU 0,5318 pada 11 gambar positif.
- Seluruh 57 unit test lulus.
- Dengan bukti ini, Tahap A selesai. Nilai tetap terbatas pada test set kecil dan tidak memvalidasi penggunaan aman.

## 8. Catatan eksekusi Tahap B

Eksekusi 5 September 2026:

- `models/baseline_manifest.json` mem-pin object checkpoint dari `ultralytics/assets` release `v8.4.0` dan tactile checkpoint dari GuideTWSI commit `58471d5ebdd574562d3dbcaf54b8df2c93688d45`.
- Hash dan ukuran kedua file remote cocok dengan artefak lokal. Loader menolak path, ukuran, SHA-256, task, atau raw labels yang berbeda sebelum inferensi.
- Model card ada di `artifacts/BASELINE_MODEL_CARD.md`. Raw label tactile `braille` dan konfigurasi upstream `tactile_paving` tetap dicatat sebagai mismatch yang belum terselesaikan.
- Object baseline dibatasi pada 80 kelas COCO; kelas khusus bahaya stasiun tidak diklaim tersedia.
- Scope dokumen aktif diselaraskan. Plan/spec lama diberi label catatan historis melalui `docs/superpowers/README.md`.
- Batch aktual: 17/17 inferensi berhasil dan seluruh status tetap STOP. Ground truth telah direview dan evaluator resmi lulus.

## 9. Catatan eksekusi awal Tahap C

Eksekusi 5 September 2026:

- `scripts/build_training_manifest.py` membuat manifest intake versioned dan menyimpan metadata lokasi, sesi, konteks, sumber/izin, kondisi kamera, task, serta SHA-256.
- Split deterministik 70/15/15 memakai grup `location_id + session_id` dan seed; frame satu sesi tidak diacak lintas split.
- Exact duplicate ditolak. Kandidat near-duplicate memakai dHash-64; kandidat yang jatuh ke split berbeda juga ditolak.
- Semua hash file aktif di `data/samples` diproteksi agar 17 foto test tidak dapat masuk intake training.
- Template, petunjuk, dan taxonomy draft tersedia di `data/training/`. Belum ada foto training baru atau taxonomy yang disetujui manusia, jadi Tahap C belum selesai.
- `scripts/training_label_gate.py` menolak kelas asing, box/polygon di luar gambar, task tidak cocok, hash test, review hilang/ambiguous, review basi setelah anotasi berubah, dan taxonomy yang belum disetujui.
- Perintah `render` menghasilkan overlay gambar asli. Perintah `review` mengikat hash anotasi aktif; perubahan berikutnya memaksa review ulang. Export YOLO memakai staging dan menghapus output parsial saat gagal.
