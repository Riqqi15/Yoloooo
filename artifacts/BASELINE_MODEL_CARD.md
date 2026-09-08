# Model Card Baseline AI Camera

Status: baseline diagnostik lokal, 5 September 2026. Bukan model siap deploy dan bukan pemberi izin berjalan.

## Cakupan

Input aktif hanya lingkungan stasiun: interior, hall/concourse, peron, akses masuk/keluar, area rel untuk analisis gambar, dan trotoar yang menjadi akses langsung ke stasiun. Adegan bus/halte sebagai konteks utama tidak termasuk. Sepuluh foto aktif berperan `test_only`, bukan data training.

Setiap hasil runtime tetap `STOP_OBSTACLE`, `STOP_UNCERTAIN`, atau `STOP_FAILURE`. Tidak ada hasil yang berarti jalur aman.

## Artefak

| Peran | Artefak | Task | Raw label | SHA-256 |
|---|---|---|---|---|
| Objek umum | `models/yolo11n.pt` | detect | 80 kelas COCO | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Kandidat tactile | `models/guidetwsi/yolo11n_tactile.pt` | segment | `{0: braille}` | `79f0bb39a354dd66d367bb5b97184df626c85f91568456e099e49a8cdbfcd878` |

Kontrak mesin berada di `models/baseline_manifest.json`. Kedua loader membandingkan path, ukuran, SHA-256, task, dan raw labels sebelum inferensi.

Sumber object checkpoint adalah [Ultralytics assets release v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0). Hash file remote `yolo11n.pt` cocok dengan file lokal. Model adalah YOLO11n pretrained COCO; COCO standar memiliki 80 kelas.

Sumber tactile checkpoint adalah [GuideTWSI](https://github.com/DARoSLab/GuideTWSI) commit `58471d5ebdd574562d3dbcaf54b8df2c93688d45`. Hash file remote pada commit tersebut cocok dengan file lokal. Repositori menyatakan lisensi MIT.

Runtime verifikasi lokal memakai Ultralytics `8.4.138`, CPU, default input `320×320`, confidence `0.25`.

## Batas semantik

Checkpoint tactile hanya menghasilkan satu kelas kandidat tactile. Metadata checkpoint berbunyi `braille`, sedangkan konfigurasi upstream yang diaudit berbunyi `tactile_paving`. Perbedaan belum terselesaikan. Output tidak boleh dianggap mampu membedakan guiding bars dan warning/truncated domes. Dua kelas membutuhkan dataset, anotasi, training, dan checkpoint baru.

Object baseline hanya mengenali kelas COCO. Beberapa kelas berguna tersedia, misalnya `person`, `bicycle`, `car`, `motorcycle`, `bus`, `train`, `truck`, `bench`, `backpack`, dan `suitcase`. Kelas bahaya stasiun berikut tidak tersedia sebagai kelas khusus: `pole`, `trash_bin`, `barrier`, `stairs`, `hole`, `step_up`, `step_down`, `platform_edge`, dan `glass_door`.

Deteksi kelas `bus` oleh model COCO tidak memperluas scope dataset menjadi bus. Itu hanya kemampuan taxonomy baseline generik.

## Evaluasi

Batch terbaru: 10/10 foto berhasil diinferensikan. Semua hasil tetap fail-safe STOP. Ground truth aktif berisi 9 anotasi AI provisional dan 1 ambiguous, sehingga evaluator resmi menolak menerbitkan metrik. Angka lama telah diarsipkan dan tidak boleh dipakai sebagai bukti performa final.

Metrik resmi baru boleh dibuat setelah review manusia, source/mask hash cocok, tidak ada row ambiguous/incomplete, dan laporan prediksi cocok dengan manifest model terpin.

## Lisensi dan penggunaan

- GuideTWSI repository/weights: MIT berdasarkan repositori upstream.
- Ultralytics software dan model: batas AGPL-3.0 atau lisensi Enterprise perlu diputuskan sesuai distribusi/penggunaan.
- Verifikasi ini bukan nasihat hukum. Audit lisensi final wajib sebelum penggunaan komersial atau distribusi aplikasi.

## Pekerjaan berikut

1. Review manusia untuk 10 overlay ground truth; putuskan sampel ambiguous.
2. Bangun dataset training stasiun baru dengan provenance lokasi/sesi dan split anti-leakage.
3. Latih detector kelas bahaya prioritas dan tactile taxonomy yang disetujui.
4. Evaluasi holdout baru, export mobile, parity test, lalu benchmark HP.
