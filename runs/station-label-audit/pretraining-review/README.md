# Review sebelum training — 8 September 2026

Status: belum siap training. Audit ini tidak mengesahkan human review.

- 50 foto sumber, 10 JSON LabelMe, 24 polygon pada saat pemeriksaan ini.
- 40 foto tanpa JSON telah melalui screening AI pada contact sheet. Kandidat dan beberapa pembanding diperiksa lewat foto asli; crop detail tersedia untuk 017, 026, 048.
- Screening AI bukan sertifikasi negatif atau review manusia. Belum dibuat label kosong atau split.
- Koreksi lama dua koordinat foto 014 dari y=2228 ke y=2227 (tinggi gambar 2228) dipertahankan. Cadangan tersedia di folder induk.
- Orientasi EXIF diterapkan untuk tampilan review dan pengecekan dimensi.
- Foto asli dan sumber/lisensi dipertahankan. Tidak ada export YOLO atau training dijalankan.

## Masalah visual yang masih terbuka

1. 001_juanda: pengguna sudah mengonfirmasi tactile ada, menggantikan pernyataan negatif sebelumnya. Kini lima polygon tersimpan; tetap perlu review akhir kelengkapan.
2. 040_137394087: tiga polygon tersimpan, termasuk tambahan bagian jauh.
3. 044_77601320: enam polygon tersimpan, termasuk pad dekat panah hijau dan kaki petugas. Foto 045_77601318 adalah marka antrean.
4. 002_sudirman: polygon depan telah dirapikan. Area peron seberang masih perlu diperiksa untuk kelengkapan label.

## Hasil screening 40 foto tanpa JSON

### 3 kandidat label tambahan

| Foto | Area | Bukti |
|---|---|---|
| 017_106327604.jpg | Strip putih beralur di sisi kanan peron, bukan garis kuning dekat kereta. Pisahkan bagian terlihat di antara tiang/bangku. | [Crop](017_106327604_detail.png) |
| 026_116115102.jpg | Strip kuning bertekstur di lantai peron yang menghadap kamera, tepat di atas dinding peron. Periksa strip peron jauh secara terpisah. | [Crop](026_116115102_detail.png) |
| 048_77563734.jpg | Pad kuning bertekstur dekat mulut eskalator, sebagian di belakang orang berdiri. Jangan sertakan sepatu atau marka antrean. | [Crop](048_77563734_detail.png) |

### 3 foto ambigu — tahan dahulu

- 009_137824158.jpg dan 010_137778724.jpg: tekstur strip tepi peron belum cukup jelas untuk keputusan positif/negatif.
- 032_166526831.jpg: panel cokelat dekat kursi belum dapat dipastikan sebagai tactile; peron jauh kanan juga perlu penilaian.

### 34 kandidat negatif sementara

Tidak ditemukan tactile yang jelas pada screening, bukan bukti bahwa stasiunnya tidak memiliki tactile. Nomor berikut merujuk prefix nama foto. Perlu review manusia sebelum disahkan negatif.

- Peron/gedung/koridor: 004, 006, 007, 011, 012, 013, 015, 016, 018, 019, 020, 023, 024, 025, 028, 029, 031, 033, 034, 035, 036, 037, 038, 039.
- Marka antrean/panah, lantai terbatas atau tertutup: 005, 030, 041, 042, 043, 045, 046, 047, 049, 050.

Foto sumber dan polygon pengguna tidak diubah pada screening ini. Konfirmasi/lengkapi 017, 026, 048 dengan tactile_paving pada permukaan terlihat saja; selesaikan tiga foto ambigu sebelum persiapan training.

## File review

- audit.json: hash sumber/anotasi, metadata sumber, dimensi, status, dan hasil validasi per foto.
- page_01.jpg sampai page_06.jpg: 50 foto dengan garis polygon yang tersimpan.
- File JPG individual bernama seperti sumber: tampilan polygon lebih besar.

Sesudah koreksi visual, jalankan ulang `rtk ./.venv/Scripts/python.exe runs/station-label-audit/prepare_review.py`. Berikutnya susun intake dengan kelompok lokasi/sesi agar foto serupa tidak bocor antar-split, konversi label ke schema training, lakukan review manusia, lalu gunakan gate validate/export yang sudah tersedia.
