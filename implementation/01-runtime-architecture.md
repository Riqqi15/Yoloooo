# Arsitektur Runtime

## Kondisi kode saat ini

Fondasi yang sudah tersedia:

- `camera` membuka kamera dan mengirim stream frame.
- `CameraGuideController` memiliki lifecycle handling dan `_busy` guard agar frame tidak menumpuk.
- ML Kit melakukan object detection lokal dengan label umum.
- Gemini Vision dipanggil berkala sebagai bantuan deskripsi jarak jauh.
- Flutter TTS memberi pesan suara.

Perubahan utama: ganti jalur keputusan keselamatan dari ML Kit/Gemini menjadi YOLO on-device. Gemini tidak boleh berada pada critical path.

## Pipeline target

```text
Camera frame + timestamp + orientation
                |
                v
        Frame quality gate
                |
      +---------+----------+
      |                    |
      v                    v
YOLO object detect   YOLO tactile segment
      |                    |
      +---------+----------+
                v
        Optional ARCore Depth
                |
                v
        Perception snapshot
                |
                v
        Hazard decision engine
                |
      +---------+----------+
      |                    |
      v                    v
  TTS/haptic        Debug overlay/log
```

## Pemisahan tanggung jawab

Rencana file mengikuti struktur feature assistant yang sudah ada:

```text
lib/features/assistant/
├── domain/
│   ├── models/camera_perception.dart
│   ├── models/camera_safety_state.dart
│   └── services/hazard_decision_engine.dart
├── data/
│   ├── services/yolo_object_detector.dart
│   ├── services/tactile_path_segmenter.dart
│   ├── services/depth_provider.dart
│   └── services/camera_guide_health_monitor.dart
└── presentation/
    ├── controllers/camera_guide_controller.dart
    └── pages/camera_guide_page.dart
```

- Controller: lifecycle, start/stop, dan publikasi state UI.
- Detector/segmenter/depth provider: membungkus runtime masing-masing.
- Decision engine: fungsi deterministik yang dapat diuji tanpa kamera.
- Health monitor: watchdog frame, latency, model, orientation, dan output channel.
- Presentation: hanya preview, overlay, status, dan emergency control.

## Model data inti

```text
PerceptionSnapshot
- frameId
- capturedAt
- processedAt
- imageQuality
- detectedObjects[]
- tactileMask?
- tactileConfidence?
- depthStatus
- depthSamples[]
- deviceOrientation
- stageTimings

SafetyDecision
- state
- priority
- messageKey
- direction
- distanceBand
- sourceSignals[]
- validUntil
```

Semua hasil memiliki timestamp. Decision engine menolak snapshot stale.

## State machine

```text
initializing
objectOnly
tactileCandidate
tactileGuidance
lostPath
degraded
uncertain
hazardStop
systemFailure
stopped
```

Aturan minimum:

- Jalur terdeteksi 3–5 frame stabil: `tactileGuidance`.
- Jalur hilang sesaat: pertahankan hasil terakhir hanya dalam jendela pendek.
- Jalur hilang 2–3 detik: `lostPath`, lalu `objectOnly` setelah STOP disampaikan.
- Bahaya kritis dari satu sumber kredibel: `hazardStop` segera.
- Bahaya dilepas hanya setelah seluruh pemeriksaan stabil.
- Kamera/model/watchdog gagal: `systemFailure`.

## Koridor berjalan

Tanpa jalur, gunakan trapesium terkalibrasi di tengah frame. Dengan jalur, koridor mengikuti mask dan centerline jalur.

Objek dianggap relevan jika:

- Footpoint atau bagian bawah mask masuk koridor.
- Arah geraknya menuju koridor.
- Jaraknya berada dalam batas peringatan.
- Termasuk kelas bahaya permukaan yang harus selalu diprioritaskan.

Objek di luar koridor tetap dapat digambar pada overlay, tetapi tidak memenuhi antrean suara biasa.

## Logika jalur

1. Ambil mask tactile yang terlihat.
2. Bagi mask menjadi beberapa band horizontal.
3. Ambil centroid setiap band.
4. Haluskan centroid antarframe.
5. Bentuk arah lurus/kiri/kanan.
6. Cari komponen jalur yang terlihat setelah objek.
7. Jangan menyambungkan gap panjang tanpa bukti.

Kondisi:

- Objek beririsan dan jalur terlihat sesudahnya: `pathBlocked`.
- Tidak ada objek, mask hilang, depth normal: `lostPath/uncertain`.
- Mask berakhir dan depth berubah: `pathEndOrDrop`.
- Dua cabang valid: STOP atau minta konteks navigasi; jangan memilih acak.

## Depth

ARCore dikonfigurasi sebagai kemampuan opsional. Runtime wajib memeriksa dukungan perangkat.

- Sampel depth di sekitar footpoint objek, bukan satu pixel.
- Gunakan median area kecil dan smoothing 3–5 frame.
- Depth invalid tidak boleh dikonversi menjadi nol meter.
- Jika tidak didukung, gunakan `near`, `medium`, dan `far` tanpa angka meter.
- Google menyatakan hasil Depth API paling akurat sekitar 0,5–5 meter dan membutuhkan gerakan perangkat. Permukaan minim fitur dapat tidak akurat: [ARCore Depth](https://developers.google.com/ar/develop/depth).

## Scheduling inference

Baseline:

```text
Object detector: setiap frame yang diterima pipeline
Tactile segmenter: setiap 2 frame
Depth: ketika tersedia dan ada area relevan
Decision engine: setiap snapshot baru
Health monitor: timer independen
```

Jika satu frame masih diproses, frame baru dibuang. Tidak ada `while (true)` yang memutar CPU.

Pada HP lambat, capability profiler menurunkan input size/rate atau beralih ke object-only. Safety tidak boleh dikurangi diam-diam.

## Feedback

Prioritas:

```text
drop/hole/platform edge
obstacle in path
lost/end path
direction correction
ordinary object
```

- Pesan kritis membatalkan TTS lama.
- Nonkritis memakai cooldown dan deduplication per tracked object.
- Bunyi/getaran STOP tidak bergantung pada kalimat TTS panjang.
- Jangan mengucapkan “aman” atau “silakan maju dengan aman”.

## UI

UI demo menampilkan:

- Camera preview.
- Bounding box dan confidence.
- Mask jalur taktil.
- Koridor berjalan.
- Mode dan safety state.
- Latency setiap tahap, total latency, serta FPS.
- Depth status.
- Pesan aktif.

Mode berpindah otomatis. Emergency stop tetap tersedia melalui navigasi Android, perintah suara, atau kontrol aksesibel yang dapat dijangkau TalkBack.

## Backend dan privasi

- Inference keselamatan lokal.
- Frame tidak disimpan di Neon.
- Frame tidak dikirim ke Gemini pada critical loop.
- Gemini dapat menjadi fitur deskripsi opsional terpisah dengan persetujuan pengguna.
- Log performa lokal tidak menyertakan gambar secara default.

## You Are Here

GPS dan kamera adalah subsistem terpisah:

- GPS memilih stasiun KRL Jabodetabek terdekat hanya jika accuracy dan radius valid.
- Map memberi label `You Are Here` pada node tersebut.
- Kamera memahami kondisi fisik depan pengguna.
- GPS tidak dipakai untuk memutuskan bahwa lantai atau jalur aman.
