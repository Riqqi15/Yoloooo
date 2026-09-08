# Evaluasi dan Keselamatan Model

## Prinsip

Accuracy total tidak cukup untuk sistem yang dapat memengaruhi keselamatan fisik. Evaluasi harus berfokus pada false negative, kondisi di luar dataset, latency, dan kemampuan sistem berhenti saat ragu.

[NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) digunakan sebagai prinsip sukarela: valid dan andal, aman, secure dan resilient, transparan, serta terus diuji.

## Metrik model

### Object detector

- Precision dan recall per kelas.
- mAP50 dan mAP50-95.
- Confusion matrix.
- False negative untuk kelas kritis.
- Recall objek yang footpoint-nya berada dalam koridor.

### Tactile segmenter

- Mask precision dan recall.
- mIoU.
- Kelanjutan centerline antarframe.
- Kesalahan jalur palsu pada lantai kuning biasa.
- Kegagalan ketika jalur tertutup sebagian.

### Depth/fusion

- Absolute error pada jarak terukur.
- Stabilitas jarak dalam 3–5 frame.
- Kemampuan mendeteksi discontinuity lantai.
- Persentase frame depth tidak valid.

## Target riset untuk demo terkontrol

Target berikut bukan sertifikasi keselamatan:

| Ukuran | Target awal |
|---|---:|
| Recall objek kritis pada local test | ≥ 0,90 |
| mIoU jalur taktil pada local test | ≥ 0,75 |
| Critical miss pada skenario demo terkontrol | 0 dari 20 pengulangan per skenario |
| Latency rata-rata setelah warm-up | < 200 ms |
| Latency p95 setelah warm-up | < 200 ms pada perangkat demo |
| Frame/keputusan stale yang dipakai | 0 |

Jika target tidak tercapai, kemampuan terkait dinonaktifkan atau diumumkan sebagai `Degraded`; threshold tidak boleh diturunkan hanya agar demo terlihat lolos.

## Safety-biased evaluation

- Untuk lubang, penurunan, tepi peron, dan penghalang di koridor, prioritaskan recall daripada precision.
- Satu sinyal bahaya yang kredibel cukup untuk menghasilkan STOP.
- Status bahaya baru boleh dilepas setelah seluruh sensor/model relevan stabil beberapa frame.
- Tidak terdeteksi tidak sama dengan aman.
- Confidence rendah, hasil bertentangan, atau scene quality buruk menghasilkan `Uncertain`, bukan “jalur aman”.

## Matriks pengujian model

| Kondisi | Object | Tactile | Depth | Hasil aman |
|---|---|---|---|---|
| Kursi tepat di jalur | terdeteksi | jalur sebelum/sesudah terlihat | valid/tidak | STOP |
| Kursi di luar koridor | terdeteksi | jalur jelas | valid | tidak mengganggu arahan utama |
| Jalur tertutup penuh | mungkin ada objek | hilang | dapat valid | STOP/Uncertain |
| Lubang menyerupai bayangan | mungkin salah | tidak relevan | anomaly/invalid | STOP jika salah satu sinyal kredibel |
| Pintu kaca berbingkai | `glass_door` | tidak relevan | mungkin tidak stabil | STOP |
| Kaca tanpa ciri | dapat terlewat | tidak relevan | dapat terlewat | residual risk tinggi; alat bantu fisik wajib |
| Kamera gelap/tertutup | hasil tidak valid | hasil tidak valid | invalid | SystemFailure + STOP |
| Frame terlambat | hasil benar tetapi stale | stale | stale | hasil dibuang + STOP |

## Threshold calibration

Jangan memakai satu confidence threshold untuk seluruh kelas.

- Kelas kritis dapat memakai threshold lebih rendah jika hasil false positive masih dapat ditangani dengan STOP.
- Kelas informasional memakai threshold lebih tinggi agar TTS tidak berisik.
- Threshold ditentukan dari precision-recall curve pada local validation.
- Simpan threshold per kelas di konfigurasi/versioned model manifest.

## Evaluasi domain perangkat

Jalankan test pada kamera berkualitas berbeda:

- HP demo utama.
- HP Android API 24+ dengan CPU/RAM rendah.
- Kamera dengan exposure buruk.
- Perangkat ARCore Depth.
- Perangkat tanpa ARCore.

Laporkan hasil per perangkat. Jangan menggabungkan angka hingga kelemahan perangkat lama tersembunyi.

## Replay test

Simpan video pengujian yang tidak mengandung identitas sensitif sebagai test fixture. Pipeline replay harus menghasilkan:

- State transition yang sama.
- Hazard priority yang sama.
- Tidak ada arahan lama setelah STOP.
- Latency yang dapat dibandingkan antarversi.

Replay test lebih aman dan konsisten daripada mengulang bahaya fisik.

## Model card wajib

`MODEL_CARD.md` berisi:

- Tujuan dan hal yang tidak boleh dilakukan model.
- Sumber serta lisensi dataset.
- Distribusi kelas.
- Perangkat dan kamera pengujian.
- Metrik per kelas.
- Failure cases yang diketahui.
- Dukungan depth.
- Input size dan quantization.
- Versi Ultralytics/runtime.
- Threshold per kelas.
- Residual risk.
- Tanggal dan orang yang menyetujui model untuk demo.

## Gate sebelum demo

Model baru hanya dipakai jika:

1. Checksum dan metadata benar.
2. Tidak ada class-index mismatch.
3. Test split dan failure set selesai.
4. Scripted demo tidak memiliki critical miss.
5. Benchmark perangkat memenuhi capability profile.
6. Watchdog dan STOP tetap bekerja ketika model sengaja dibuat gagal.
7. Risk register diperbarui.

Gate ini hanya memberi izin demo terkontrol, bukan penggunaan mandiri di stasiun nyata.
