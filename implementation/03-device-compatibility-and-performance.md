# Kompatibilitas Perangkat dan Performa

## Kenyataan perangkat lama

Jika “HP tahun 2010-an” berarti perangkat yang benar-benar dirilis sekitar 2010–2014, banyak perangkat tidak dapat menjalankan aplikasi Flutter modern. Dokumentasi Flutter 3.44.7 mendukung Android API 24–37 dan menyatakan API 23 ke bawah tidak didukung: [Flutter supported platforms](https://docs.flutter.dev/reference/supported-platforms).

Android API 24 adalah Android 7.0. Perangkat yang tidak dapat menjalankan API 24 tidak masuk target APK saat ini.

ARCore juga tidak tersedia pada semua perangkat. Aplikasi harus dikonfigurasi sebagai **AR Optional** dan memeriksa dukungan saat runtime: [ARCore optional apps](https://developers.google.com/ar/develop/c/enable-arcore).

## Capability profile

Jangan memilih fitur hanya dari nama HP atau jumlah RAM. Jalankan startup benchmark setelah model warm-up.

| Profil | Kondisi | Kemampuan |
|---|---|---|
| Unsupported | Android API < 24 atau model tidak dapat load | AI Camera dinonaktifkan |
| Basic | model object 320 dapat berjalan, tanpa depth | object awareness depan; tanpa meter dan tanpa active path guidance |
| Balanced | object + tactile memenuhi batas bergantian | object + jalur, distance band, tanpa depth metrik |
| Full | object + tactile + ARCore Depth stabil | seluruh fitur prototipe |
| Degraded | thermal/latency memburuk saat runtime | turunkan rate; jika tetap lambat, STOP |

Pengguna harus diberi tahu profil aktif. Fitur tidak boleh hilang diam-diam.

## Startup benchmark

1. Periksa Android API dan ABI.
2. Muat model dan validasi checksum.
3. Jalankan 10–20 warm-up inference.
4. Jalankan sekurangnya 30 inference object.
5. Jalankan sekurangnya 30 inference tactile.
6. Ukur mean, p50, p95, max, dan peak RAM.
7. Periksa GPU/NNAPI delegate.
8. Periksa ARCore dan Depth support.
9. Pilih capability profile.

Profil dapat diturunkan ketika thermal throttling atau latency buruk, tetapi tidak dinaikkan lagi sampai sistem stabil dan pengguna tidak sedang berjalan.

## Target latency

```text
capture timestamp
-> preprocessing
-> inference
-> fusion
-> safety decision
-> TTS/haptic dispatch
```

Target p95 setelah warm-up: di bawah 200 ms pada perangkat demo.

Durasi suara sampai selesai tidak dihitung. Waktu sampai perintah audio/haptic dikirim tetap dihitung.

## Strategi optimasi

- YOLO varian nano.
- Model LiteRT/TFLite terkuantisasi.
- Input 320×320 sebagai baseline.
- Satu frame in-flight; frame berikutnya dibuang.
- Object detector setiap frame pipeline.
- Tactile segmenter setiap dua frame dan hasilnya dilacak singkat.
- Depth hanya disampel pada koridor/footpoint relevan.
- Reuse tensor/buffer.
- Load model satu kali.
- Build `release`, bukan `debug`, untuk benchmark.
- Hindari encoding JPEG pada critical path.
- Gemini/backend tidak berada pada loop 200 ms.

## Runtime Flutter

Pilihan pertama: [plugin resmi Ultralytics YOLO Flutter](https://pub.dev/packages/ultralytics_yolo), karena mendukung detection, segmentation, custom LiteRT model, real-time camera, result streaming, dan accelerator fallback.

Catatan:

- Lisensi plugin/model Ultralytics adalah AGPL-3.0 atau enterprise.
- Model official umumnya memakai input 640; untuk HP lama gunakan custom export 320.
- Benchmark tetap wajib karena dukungan GPU bergantung driver perangkat.
- Jika dua model/camera view tidak dapat dijadwalkan sesuai kebutuhan safety, gunakan single-image inference dari frame stream atau native LiteRT bridge minimal.

## RAM, bukan storage

RAM dipakai oleh:

- Camera buffers.
- Dua model.
- Input/output tensors.
- Segmentation masks.
- Depth maps.
- Flutter UI dan TTS.

Masalah utama bukan ukuran database. Risiko muncul jika frame/tensor disimpan tanpa dilepas atau beberapa instance model dibuat.

Mitigasi:

- Backpressure.
- Reuse buffer.
- Jangan menyimpan `CameraImage` ke list.
- Dispose model dan camera pada lifecycle stop.
- Batasi riwayat mask/depth ke beberapa frame.
- Pantau peak RSS/Java/native heap dengan Android Studio Profiler atau `adb shell dumpsys meminfo`.

## Kebijakan HP lambat

```text
p95 <= 200 ms
-> profil terkait boleh aktif

200 ms < p95 <= 500 ms
-> object awareness terbatas; tidak memberi active walking guidance

p95 > 500 ms atau frame watchdog gagal
-> AI Camera unavailable + STOP
```

Angka kedua adalah fail-safe engineering awal dan harus dikalibrasi pada pengujian. Jangan memaksakan dua model pada perangkat yang tidak mampu.

## Depth fallback

- Full: ARCore Depth dan smoothing.
- Balanced: depth tidak tersedia; gunakan near/medium/far dari heuristik yang telah dikalibrasi dan jangan menyebut meter.
- Basic: hanya arah dan jenis objek; jangan menyimpulkan lubang/tinggi tanpa bukti.
- Unsupported: fitur kamera dinonaktifkan.

Depth API paling akurat sekitar 0,5–5 meter, membutuhkan gerakan perangkat, dan dapat tidak akurat pada permukaan minim tekstur. Ini sesuai target 5 meter tetapi bukan jaminan ketepatan.

## Matriks perangkat pengujian

Minimal:

1. Emulator hanya untuk UI dan unit test.
2. HP demo utama yang mendukung ARCore Depth.
3. HP API 24+ tanpa ARCore.
4. HP low-end dengan RAM/CPU rendah.
5. Satu perangkat dengan kualitas kamera buruk jika tersedia.

Catat model HP, Android API, ABI, RAM, delegate, input size, latency, FPS, thermal state, dan battery drain.
