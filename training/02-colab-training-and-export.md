# Training Colab dan Export Model

## Lingkungan

Gunakan Google Colab GPU untuk training. Pin versi dependency dalam notebook final agar hasil dapat direproduksi.

Notebook runnable untuk baseline tactile satu kelas tersedia di `notebooks/train_tactile_one_class_colab.ipynb`. Notebook tersebut mem-pin Ultralytics 8.4.138 dan menggunakan label tunggal `tactile_paving`. Target terpisah `tactile_guiding_path`/`tactile_warning_block` ditunda sampai tersedia data yang dilabeli khusus untuk kedua kelas.

```bash
pip install ultralytics huggingface_hub
nvidia-smi
```

Simpan:

- Notebook.
- `requirements` atau hasil `pip freeze` terpilih.
- Dataset version.
- Seed.
- Command training.
- Checkpoint terbaik.
- Hasil evaluasi.

## Download GuideTWSI

```bash
huggingface-cli download guidedogrobot-tactile/GuideTWSI \
  --repo-type dataset \
  --local-dir ./data/guidetwsi

huggingface-cli download guidedogrobot-tactile/GuideTWSI-weights \
  --local-dir ./checkpoints/guidetwsi
```

Pretrained utama: `yolo11n_tactile.pt`.

## Struktur dataset object detection

```text
data/objects/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── objects.yaml
```

Contoh `objects.yaml`:

```yaml
path: /content/data/objects
train: images/train
val: images/val
test: images/test
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

## Struktur dataset segmentation

```text
data/tactile/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── tactile.yaml
```

Label segmentation mengikuti format polygon YOLO. Pastikan metadata class tertanam saat export agar runtime tidak salah membaca indeks.

## Training object detector

Baseline:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=/content/data/objects/objects.yaml \
  epochs=100 \
  imgsz=640 \
  batch=-1 \
  device=0 \
  seed=42 \
  project=/content/runs/objects \
  name=yolo11n_objects
```

Jangan langsung memakai epoch lebih besar untuk menutupi dataset buruk. Periksa loss, precision/recall per kelas, confusion matrix, dan contoh false negative.

## Fine-tune segmenter jalur taktil

```bash
yolo segment train \
  model=/content/checkpoints/guidetwsi/yolo11n_tactile.pt \
  data=/content/data/tactile/tactile.yaml \
  epochs=80 \
  imgsz=640 \
  batch=-1 \
  device=0 \
  seed=42 \
  project=/content/runs/tactile \
  name=yolo11n_tactile_local
```

Jika pretrained tidak kompatibel dengan versi Ultralytics yang dipin, gunakan konfigurasi/notebook resmi GuideTWSI untuk menghasilkan checkpoint baru sebelum fine-tuning lokal.

## Augmentasi

Gunakan augmentasi yang merepresentasikan kamera berjalan:

- Brightness/exposure ringan.
- Motion blur ringan.
- Perspective dan rotation kecil.
- Noise kamera.
- Shadow atau partial occlusion.

Hindari augmentasi yang mengubah bentuk lubang, menghapus tekstur jalur, atau menghasilkan skenario fisik tidak masuk akal. Warna jalur bukan satu-satunya ciri; model harus belajar tekstur dan bentuk.

## Validasi checkpoint

```bash
yolo detect val \
  model=/content/runs/objects/yolo11n_objects/weights/best.pt \
  data=/content/data/objects/objects.yaml \
  imgsz=640

yolo segment val \
  model=/content/runs/tactile/yolo11n_tactile_local/weights/best.pt \
  data=/content/data/tactile/tactile.yaml \
  imgsz=640
```

Selain test split, jalankan inference pada `failure_test` stasiun dan replay stasiun yang tidak pernah masuk training. Rekaman indoor non-stasiun hanya boleh dipakai untuk uji fungsi terkontrol, bukan metrik domain stasiun.

## Export untuk Android

Target pertama: input 320×320 agar realistis untuk HP menengah dan lama. Bandingkan dengan 416×416 jika objek kecil terlalu sering terlewat.

Dengan Ultralytics versi yang mendukung LiteRT:

```python
from ultralytics import YOLO

objects = YOLO("/content/runs/objects/yolo11n_objects/weights/best.pt")
objects.export(
    format="litert",
    imgsz=320,
    quantize=8,
    data="/content/data/objects/objects.yaml",
    nms=False,
    end2end=False,
)

tactile = YOLO("/content/runs/tactile/yolo11n_tactile_local/weights/best.pt")
tactile.export(
    format="litert",
    imgsz=320,
    quantize=8,
    data="/content/data/tactile/tactile.yaml",
    nms=False,
    end2end=False,
)
```

Nama argumen export dapat berubah antarversi Ultralytics. Notebook final harus mem-pin versi yang telah berhasil dan merujuk [dokumentasi export resmi](https://docs.ultralytics.com/modes/export/).

## Verifikasi hasil export

Untuk setiap file `.tflite`/LiteRT:

1. Pastikan file dapat dimuat ulang.
2. Jalankan test image yang sama pada `.pt` dan hasil export.
3. Bandingkan class ID, confidence, bounding box, dan mask.
4. Pastikan metadata task dan label tertanam.
5. Catat ukuran file dan peak RAM runtime.
6. Jalankan minimal satu inference sebelum benchmark sebagai warm-up.
7. Hitung checksum SHA-256 dan simpan di model manifest aplikasi.

## Artefak akhir training

```text
artifacts/
├── object_detector_320_int8.tflite
├── tactile_segmenter_320_int8.tflite
├── labels_objects.txt
├── labels_tactile.txt
├── model_manifest.json
├── metrics_objects.json
├── metrics_tactile.json
├── confusion_matrix_objects.png
├── failure_samples/
└── MODEL_CARD.md
```

Model tidak boleh dipasang ke APK hanya karena training selesai. Model harus melewati evaluasi keselamatan pada dokumen berikutnya.

Bundle hasil notebook diverifikasi setelah diunduh:

```powershell
rtk .\.venv\Scripts\python.exe scripts\verify_tactile_candidate.py artifacts\candidates\tactile-one-class-v1
```

Verifier memeriksa file wajib, SHA-256, konfigurasi training, task segmentation, dan label satu kelas. Hasil `candidate_valid` belum menjadikan model siap deployment.
