# Demo, Pengujian, dan Observability

## Tujuan demo

Latihan indoor terkontrol (misalnya ruang meeting) hanya membuktikan fungsi runtime berikut, bukan performa domain stasiun:

- Sistem mendeteksi objek dalam koridor depan.
- Objek samping tidak membanjiri suara.
- Jalur taktil muncul otomatis ketika tersedia.
- Penghalang di atas jalur menghasilkan STOP.
- Jalur hilang tidak membuat sistem menebak.
- Safety watchdog bekerja ketika kamera/model gagal.
- Latency setiap proses terlihat dan target 200 ms dapat diperiksa.

Demo tidak membuktikan bahwa sistem aman digunakan mandiri di stasiun.

## Peralatan

- HP fisik dengan build release.
- Harness dada/pinggang yang stabil.
- Open-ear speaker/headphone atau speaker HP.
- Karpet/papan tactile kuning; pita kuning polos hanya untuk uji visual awal.
- Kursi, meja, tas, tempat sampah bersih, dan pembatas lunak.
- Laptop untuk membaca log.
- Pengawas yang dapat melihat.

## Skenario

| ID | Kondisi | Hasil yang diharapkan |
|---|---|---|
| D01 | Tidak ada jalur, kursi di samping | tidak ada peringatan utama |
| D02 | Tidak ada jalur, kursi di tengah | objek + arah + distance band |
| D03 | Orang melintas ke koridor | warning diprioritaskan |
| D04 | Jalur tactile bersih | mode tactile aktif stabil |
| D05 | Kursi menutup jalur | STOP; jalur setelah objek boleh diinformasikan |
| D06 | Tempat sampah menutup jalur | STOP dengan label benar atau generic obstacle |
| D07 | Jalur hilang | STOP/uncertain, tidak menebak arah |
| D08 | Jalur berakhir | STOP dan status path end |
| D09 | Kamera ditutup | watchdog + systemFailure + STOP |
| D10 | Kamera terlalu miring | orientation failure + STOP |
| D11 | Cahaya diredupkan | degraded/uncertain + STOP |
| D12 | Model sengaja gagal load | panduan tidak mulai |
| D13 | TTS queue berisi pesan lama lalu bahaya muncul | pesan lama dibatalkan, STOP berbunyi |
| D14 | Benchmark 1–5 menit | tidak ada frame backlog; p95 dilaporkan |

Lubang, tangga turun, dan tepi tidak dibuat secara fisik. Gunakan replay video, mock visual aman, atau lokasi yang dibatasi tanpa pengguna berjalan.

## Continuous processing

Gunakan camera stream dengan backpressure:

```text
frame arrives
if processing: increment droppedFrames, discard
else: process and publish snapshot
```

Tidak menggunakan busy `while (true)`. Watchdog memakai timer independen agar tetap dapat mendeteksi pipeline yang macet.

## Stage timing

Setiap snapshot mencatat:

```text
capture_to_start_ms
preprocess_ms
object_inference_ms
tactile_inference_ms
depth_ms
fusion_ms
decision_ms
feedback_dispatch_ms
total_ms
```

Overlay debug:

```text
Mode: TACTILE_GUIDANCE
Safety: SAFE_TO_ANALYZE
Object: 54 ms
Tactile: 68 ms
Depth: 17 ms
Decision: 3 ms
Total: 158 ms
FPS: 6.3
Dropped: 42
Profile: FULL
```

## CSV lokal

Header yang disarankan:

```csv
timestamp,frame_id,device_profile,safety_state,object_ms,tactile_ms,depth_ms,decision_ms,total_ms,fps,dropped_frames,object_count,path_confidence,depth_status,message_key
```

- Tidak menyimpan image bytes.
- Tidak menyimpan token/auth.
- Lokasi presisi tidak masuk log benchmark.
- File dapat dihapus setelah evaluasi.

## Unit test decision engine

Kasus minimum:

- Objek luar koridor tidak menghasilkan warning.
- Objek masuk koridor menghasilkan warning.
- Hole signal tunggal menghasilkan STOP.
- Depth drop tunggal menghasilkan STOP.
- Hasil stale ditolak.
- Jalur stabil mengaktifkan tactile mode.
- Jalur hilang singkat tidak menyebabkan jitter.
- Jalur hilang melewati timeout menghasilkan STOP.
- Hazard baru membatalkan message lama.
- System failure selalu lebih tinggi daripada arahan arah.

Test memakai object/mask/depth sintetis. Tidak perlu kamera.

## Replay integration test

Gunakan video tetap untuk membandingkan versi model dan decision engine. Simpan expected timeline:

```text
00:00 initializing
00:02 objectOnly
00:05 tactileCandidate
00:06 tactileGuidance
00:12 hazardStop(chair)
00:18 uncertain
```

Perubahan timeline harus direview. Snapshot gambar failure boleh disimpan hanya dalam dataset development dengan izin dan aturan privasi.

## Benchmark protocol

1. Install clean release APK.
2. Reboot HP atau hentikan aplikasi berat.
3. Catat suhu dan baterai awal.
4. Jalankan startup warm-up.
5. Jalankan skenario yang sama minimal 1 menit; uji thermal 5–10 menit bila memungkinkan.
6. Export CSV.
7. Hitung mean, p50, p95, max, FPS, dan drop ratio.
8. Catat suhu/baterai akhir dan delegate yang dipakai.
9. Ulangi pada tiap capability profile.

## Acceptance demo

- Semua skenario kritis menghasilkan STOP pada 20 pengulangan terkontrol.
- Tidak ada arahan “aman” atau arahan lama setelah STOP.
- Tidak ada frame stale yang dipakai.
- p95 total di bawah 200 ms untuk profile yang diaktifkan pada HP demo.
- Kamera tetap responsif dan RAM stabil.
- Menutup kamera atau mematikan model menghasilkan fail-safe.
- Perangkat tanpa depth tidak menyebut angka meter.
- Debug overlay dan CSV menunjukkan waktu setiap proses.

## Pelaporan bug safety

Setiap kegagalan mencatat:

- Model/version manifest.
- Device/profile.
- Scenario ID.
- Expected dan actual safety state.
- Timestamp timeline.
- Apakah terjadi false negative, false positive, stale result, atau output failure.
- Mitigasi sebelum demo berikutnya.

Critical false negative menghentikan penggunaan model sampai penyebabnya dipahami dan gate diulang.
