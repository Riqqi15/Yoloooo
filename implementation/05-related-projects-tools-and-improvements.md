# Proyek Serupa, Tools, dan Improvisasi

Referensi diperiksa pada 2 September 2026. Kode pihak ketiga menjadi bahan belajar, bukan otomatis layak disalin ke critical safety path.

## Proyek serupa

### 1. Google Research Project Guideline

[Repository](https://github.com/google-research/project-guideline)

Proyek paling dekat dengan kebutuhan:

- On-device ML untuk pengguna tunanetra.
- Segmentasi garis fisik.
- Depth map dan ARCore motion tracking.
- Occupancy map untuk penghalang.
- Audio navigation berlatensi rendah.
- STOP ketika tracking hilang, anomaly muncul, atau navigasi tidak tersedia.
- Logging dan simulator untuk replay.

Hal yang diadopsi:

- Tracking state eksplisit.
- Fail-safe STOP.
- Clearance corridor/occupancy concept.
- Pemisahan detector, guidance, control, audio, visualization, dan logging.
- Simulator/replay sebelum pengujian fisik.
- Sighted spotter selama penggunaan riset.

Keterbatasan penting: dokumentasinya mensyaratkan Pixel 6, 7, atau 8 dengan Android 11+ dan TPU. Ini bukti bahwa full navigation tidak realistis dipaksakan ke semua HP lama.

### 2. GuideTWSI

[Repository](https://github.com/DARoSLab/GuideTWSI)

- Dataset 39,5 ribu gambar jalur taktil nyata dan sintetis.
- Pretrained `yolo11n_tactile.pt`.
- Notebook YOLO11-Seg dan beberapa baseline segmentation.
- Evaluasi precision, recall, mAP, dan mIoU.

Hal yang diadopsi:

- Pretrained segmenter sebagai baseline.
- Synthetic augmentation untuk variasi jalur.
- Fine-tune data lokal dan evaluasi lintas-domain.

### 3. ROD: Real-Time Obstacle Detection

[Dataset](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO)

- 24.326 gambar dan 25 kelas.
- Fokus smartphone assistive vision.
- Menggabungkan YOLOv8n, ARCore, TTS, dan haptic.
- Memuat objek yang sering tidak ada di COCO: dustbin, electrical pole, manhole, stairs, dan street furniture.

Hal yang diadopsi:

- Taxonomy penghalang pedestrian.
- YOLO nano untuk perangkat konsumen.
- Kombinasi deteksi, estimasi jarak, TTS, dan getaran.

Audit lisensi tetap diperlukan karena dataset menggabungkan banyak sumber Roboflow.

### 4. ARCore Depth Lab dan raw depth sample

- [ARCore Depth Lab](https://github.com/googlesamples/arcore-depth-lab)
- [ARCore Android raw depth sample](https://github.com/google-ar/arcore-android-sdk/tree/main/samples/raw_depth_java)

Hal yang dipelajari:

- Depth image lifecycle.
- Depth/point-cloud visualization.
- Collision dan geometry-aware processing.
- Runtime tracking checks.

Depth Lab tidak lagi aktif dipelihara dan berbasis Unity; untuk implementasi Android gunakan dokumentasi serta sample raw depth Android terbaru sebagai sumber utama.

### 5. GuideLens Android

[Repository](https://github.com/nrai18/GuideLens-Android-App)

Proyek komunitas Android yang menggabungkan:

- YOLO INT8 object detection.
- Semantic floor segmentation.
- Grid pathfinding.
- Filtering/hysteresis agar arah tidak jitter.
- TTS on-device.

Hal yang berguna: adaptive configuration, low-pass filtering, dan pemisahan walkable floor dari object detector.

Catatan: klaim “100% failsafe” tidak boleh diadopsi. Tidak ada model vision yang dapat menjamin seluruh kegagalan tertangani. Angka performa juga harus diuji ulang pada perangkat kita.

### 6. Blind's Eye

[Repository](https://github.com/adidev001/blinds_eye)

Proyek wearable berbasis Python yang memakai YOLOv11, Depth Anything V2, offline TTS, serta mode indoor/outdoor.

Hal yang diadopsi:

- Context-aware class filtering untuk mengurangi audio noise.
- Direction `left/center/right`.
- Mode tanpa depth.

Jangan mengadopsi formula jarak monocular sederhana sebagai jarak keselamatan tanpa kalibrasi kamera dan validasi lapangan.

## Tools yang direkomendasikan

| Tahap | Tool | Alasan |
|---|---|---|
| Labeling | [CVAT](https://www.cvat.ai/) atau [Roboflow](https://roboflow.com/) | box dan polygon/mask |
| Training | Google Colab | GPU gratis/terbatas dan notebook mudah dibagikan |
| Model | [Ultralytics YOLO](https://docs.ultralytics.com/) | detect, segment, export LiteRT |
| Dataset hosting | Google Drive, Kaggle, atau Hugging Face | jangan pakai Neon untuk gambar |
| Flutter runtime | [ultralytics_yolo](https://pub.dev/packages/ultralytics_yolo) | plugin resmi, custom TFLite, live results |
| Depth | [ARCore Android SDK](https://github.com/google-ar/arcore-android-sdk) | runtime support check dan raw depth |
| Profiling | Flutter DevTools + Android Studio Profiler | frame time, CPU, native/Java RAM |
| System log | CSV lokal + replay timeline | mudah dianalisis tanpa menyimpan frame |
| Risk management | [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | struktur validitas, safety, resilience, transparency |

## Improvisasi yang direkomendasikan

### 1. Automatic capability profiler

Jalankan benchmark saat setup dan pilih Basic/Balanced/Full. Lebih akurat daripada menebak kemampuan dari tahun HP atau RAM.

### 2. Stop cue terpisah dari TTS

Gunakan bunyi dan getaran STOP yang sudah dimuat di memori. Bahaya tidak menunggu engine TTS menyelesaikan kalimat.

### 3. Audio spatial sederhana

Selain kalimat, gunakan bunyi stereo kiri/kanan untuk koreksi kecil. TTS dipakai untuk objek dan bahaya. Pola ini terinspirasi Project Guideline dan mengurangi beban kognitif.

### 4. Risk-aware scheduling

Object detector tetap sering berjalan. Segmenter dan depth diberi prioritas lebih tinggi ketika jalur atau ground anomaly muncul. Model tidak harus selalu berjalan pada frekuensi sama.

### 5. Scene quality gate

Hitung blur, luminance, contrast, frame freeze, dan orientation sebelum inference. Model tidak boleh memberikan arahan dari frame yang secara visual sudah tidak layak.

### 6. Replay harness

Rekam fixture pengembangan yang aman dan jalankan ulang setiap perubahan model/decision engine. Hasil state timeline dapat dibandingkan otomatis.

### 7. Active learning lokal

Simpan metadata kegagalan dan hanya simpan frame setelah persetujuan developer/peserta. Tambahkan false negative ke dataset berikutnya. Tidak ada upload otomatis.

### 8. Sensor eksternal opsional

Untuk penelitian lanjutan, ultrasonic atau ToF wearable membantu memberi bukti keberadaan penghalang yang sulit terlihat. Sensor ini tetap bukan jaminan untuk kabel tipis atau seluruh kaca.

### 9. Environmental mode tanpa tombol pengguna

Sistem dapat memilih konteks otomatis:

- No tactile path: object corridor mode.
- Tactile path stable: tactile guidance mode.
- Scene invalid: STOP.

Tidak ada toggle mode manual. Developer overlay tetap dapat menunjukkan keputusan sistem.

### 10. Model manifest dan rollback

Setiap model membawa version, checksum, labels, thresholds, input size, dataset version, dan metrics. APK menyimpan satu model terakhir yang telah lolos gate agar update gagal dapat di-rollback.

## Keputusan arsitektur akhir

- Dua model kecil lebih mudah diukur dan dijadwalkan daripada satu model besar.
- ARCore optional, bukan syarat instalasi seluruh aplikasi.
- HP literal lama yang tidak mendukung API 24 tidak dapat menjalankan build Flutter saat ini.
- HP API 24+ lambat mendapat object-only atau degraded mode; active guidance dinonaktifkan jika latency tidak aman.
- Project Guideline menjadi referensi keselamatan dan state machine terkuat.
- GuideTWSI menjadi baseline jalur taktil.
- ROD menjadi baseline objek pedestrian.
- Kamera saja tidak dapat menjamin kaca bersih, kabel tipis, atau seluruh drop-off; residual risk dan alat bantu fisik tetap wajib.
