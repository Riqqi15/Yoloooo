# Dataset dan Labeling

## Tujuan

Menghasilkan dua dataset terpisah agar model kecil, mudah diuji, dan dapat dijadwalkan berbeda pada HP lambat:

1. Dataset object detection untuk benda dan bahaya.
2. Dataset instance segmentation untuk jalur taktil.

Jangan memaksa seluruh tugas menjadi satu model. Format label, kebutuhan keselamatan, dan interval inference berbeda.

## Sumber data awal

Scope proyek: lingkungan stasiun dan trotoar umum yang menjadi akses atau terhubung langsung dengan stasiun. Foto bus/halte sebagai konteks utama dan trotoar tanpa konteks stasiun dikecualikan. Dataset publik di bawah adalah calon referensi; setiap gambar perlu disaring sebelum dimasukkan ke dataset proyek. Registry test aktif berada di `data/dataset_scope.json`.

| Sumber | Kegunaan | Format/lisensi | Catatan |
|---|---|---|---|
| [GuideTWSI](https://github.com/DARoSLab/GuideTWSI) | Mask jalur taktil directional bars dan warning domes | Segmentasi, MIT | 39,5 ribu gambar dan pretrained YOLO11n-Seg tersedia |
| [ROD](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO) | Orang dan 25 objek/bahaya pedestrian | YOLO, wrapper MIT | 24.326 gambar; sumber Roboflow mempertahankan lisensi asal |
| [WOTR](https://github.com/kxzr/WOTR) | Jalur taktil, orang, tiang, pembatas, kendaraan | Pascal VOC, MIT | Perlu konversi VOC ke YOLO dan audit nama kelas `blind_road` |
| [Blind Navigation Obstacle](https://huggingface.co/datasets/ALTHISM/Obstacle) | Hambatan untuk navigasi tunanetra | YOLO, CC BY-NC 4.0 | Gated dan nonkomersial |
| [Mendeley obstacle dataset](https://data.mendeley.com/datasets/xwhnp82rhk/1) | Pole, fence, bump, hole | YOLO, CC BY 4.0 | Berguna memperkaya kelas bahaya permukaan |
| [Mendeley navigation dataset](https://data.mendeley.com/datasets/m68g3h7p87/1) | Objek dari perspektif pedestrian | Detection/segmentation, CC BY 4.0 | 22 kelas dan lebih dari 8 ribu gambar |
| [Roboflow tactile paving](https://universe.roboflow.com/tactile-paving-detection/tactile_paving/dataset/1) | Eksperimen cepat jalur taktil | YOLO, CC BY 4.0 | Hanya sekitar 438 gambar; jangan dijadikan satu-satunya sumber |

## Skema label object detection

Skema awal yang dipakai aplikasi:

```yaml
names:
  0: person
  1: chair
  2: table
  3: pole
  4: trash_bin
  5: luggage
  6: backpack
  7: barrier
  8: bench
  9: bicycle
  10: motorcycle
  11: vehicle
  12: stairs
  13: hole
  14: step_up
  15: step_down
  16: platform_edge
  17: glass_door
```

Kelas dapat dikurangi setelah audit distribusi data. Jangan mempertahankan kelas yang hanya memiliki sedikit contoh dan tidak dapat diuji secara aman. `step_up`, `step_down`, dan `platform_edge` harus diperkuat dengan depth; bounding box bukan bukti tunggal.

## Skema label jalur taktil

Gunakan polygon/mask, bukan bounding box:

```yaml
names:
  0: tactile_guiding_path
  1: tactile_warning_block
```

- `tactile_guiding_path`: ubin bergaris untuk panduan arah.
- `tactile_warning_block`: ubin titik/dome sebagai peringatan berhenti atau perubahan area.
- Label hanya pixel yang terlihat. Jangan menggambar mask di balik orang atau benda.
- Kelanjutan jalur setelah penghalang dilabel sebagai komponen terlihat yang terpisah. Decision engine yang menyimpulkan hubungan keduanya.

## Aturan bounding box

- Box mengikuti bagian objek yang terlihat, tidak memasukkan bayangan.
- Bagian bawah objek harus akurat karena dipakai sebagai `footpoint` untuk menguji irisan dengan koridor berjalan.
- Orang yang terpotong frame tetap dilabel jika tubuhnya dapat dikenali.
- Lubang dilabel pada bukaan/area permukaan yang berbahaya, bukan seluruh lantai gelap.
- Pintu kaca dilabel ketika petunjuk fisiknya terlihat: bingkai, gagang, sambungan, stiker, atau pantulan.
- Kabel tipis dicatat, tetapi tidak boleh dijanjikan selalu terdeteksi oleh kamera HP.

## Data lokal wajib

Data publik tidak mewakili seluruh kondisi stasiun Indonesia dan trotoar aksesnya. Rekam dari posisi HP di dada atau pinggang:

- Jalur lurus, melengkung, bercabang, berakhir, dan terputus.
- Jalur bersih serta tertutup orang, kursi, tas, tiang, dan tempat sampah.
- Objek sama di atas jalur dan di luar koridor.
- Lantai kuning yang bukan jalur taktil sebagai negative samples.
- Lantai berpola, reflektif, basah, redup, silau, dan terkena bayangan.
- Kamera stabil, berguncang, miring, terlalu atas, dan terlalu bawah.
- Pintu kaca dengan bingkai, gagang, stiker, refleksi, serta kaca tanpa petunjuk kuat.
- Foto tanpa jalur dan tanpa objek penting.
- HP/kamera berbeda jika tersedia.

Mulai dari beberapa ratus frame unik per domain. Tambah data berdasarkan kesalahan nyata, bukan menambah augmentasi tanpa arah.

## Pengambilan data aman

- Pengambil data harus dapat melihat dan tidak berjalan menuju bahaya nyata.
- Jangan membuat lubang, tepi, atau kabel berbahaya untuk kebutuhan rekaman.
- Gunakan replika aman, rekaman publik berlisensi, atau area yang dibatasi.
- Jangan merekam wajah dan identitas tanpa persetujuan.
- Simpan data training di penyimpanan dataset, bukan Neon produksi.

## Split dataset

Gunakan split berdasarkan sesi rekaman atau lokasi:

```text
train 70%
validation 15%
test 15%
```

Frame berdekatan dari satu video tidak boleh tersebar ke train dan test. Jika tersebar, hasil evaluasi terlihat tinggi tetapi tidak mengukur generalisasi.

Tujuh belas gambar aktif di `data/samples` sudah ditetapkan sebagai `test_only`; jangan gunakan untuk training atau pemilihan hyperparameter. Anotasinya telah direview Riyadh. Status lolos validator tetap hanya membuktikan struktur dan integritas data, bukan keselamatan penggunaan.

Jalur intake lokal tersedia di `data/training/` dan dibangun dengan:

```powershell
rtk .\.venv\Scripts\python.exe scripts\build_training_manifest.py --dataset-version station-v1
```

Builder mewajibkan provenance, hash konten, lokasi/sesi, konteks, izin penggunaan, serta kondisi pengambilan. Exact duplicate dan hash milik 17 test aktif ditolak. Kandidat near-duplicate dHash tidak boleh melintasi split. Split 70/15/15 ditentukan dari grup `location_id + session_id` dengan seed tersimpan.

Sesudah anotasi, gunakan `scripts/training_label_gate.py`. Tool membuat overlay QA, mengikat persetujuan manusia ke hash sumber+anotasi, memvalidasi box/polygon di koordinat asli, melaporkan distribusi kelas/split/kondisi, dan mengekspor YOLO secara atomik. Label `unreviewed` atau `ambiguous` tidak dapat diekspor.

Sediakan `failure_test` terpisah untuk:

- Gelap dan silau.
- Motion blur.
- Jalur tertutup penuh.
- Kaca transparan.
- Kabel tipis.
- Lubang menyerupai bayangan.
- Perangkat dengan kualitas kamera rendah.

## Quality control label

1. Annotator pertama membuat label.
2. Reviewer memeriksa seluruh kelas kritis: `hole`, `step_down`, `platform_edge`, `glass_door`.
3. Ambil sampel minimal 10% kelas biasa untuk review kedua.
4. Jalankan pemeriksaan box kosong, polygon rusak, class ID tidak dikenal, dan gambar duplikat.
5. Simpan versi dataset dan changelog perubahan label.

## Metadata minimum

Setiap sesi memiliki metadata nonidentitas:

```text
session_id
location_type: station_interior | station_concourse | station_platform | station_track_area | station_access | station_access_sidewalk
device_model
camera_position: chest | waist | hand
lighting: bright | normal | dim | glare
motion: static | walking | shaking
surface: matte | reflective | patterned | wet
```

Metadata dipakai untuk menganalisis kegagalan, bukan sebagai input model.
