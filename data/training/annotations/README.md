# Format Anotasi Training

Satu file per gambar: `<sample_id>.json`. `sample_id` adalah SHA-256 gambar dari manifest intake.

```json
{
  "schema_version": 1,
  "sample_id": "sha256-gambar",
  "source_sha256": "sha256-gambar",
  "objects": [
    {"label": "person", "bbox": [120, 80, 300, 620]}
  ],
  "tactile": [
    {
      "label": "tactile_guiding_path",
      "points": [[100, 400], [160, 400], [300, 700], [180, 700]]
    }
  ]
}
```

Koordinat memakai pixel gambar asli. Box memakai `x1,y1,x2,y2`. Polygon hanya mengikuti pixel tactile yang terlihat; bagian tertutup dibuat sebagai polygon terpisah. Array kosong berarti negative sample dan tetap memerlukan review.

Urutan wajib:

1. Buat anotasi.
2. Jalankan `render` dan lihat setiap overlay.
3. Perbaiki koordinat yang meleset.
4. Human reviewer menjalankan `review`; jangan meminta AI mengesahkan review manusia.
5. Jalankan `validate`.
6. Jalankan `export` hanya bila `training_ready: true`.

Review mengikat hash sumber dan hash anotasi. Perubahan setelah review otomatis membuat status tidak valid sampai diperiksa ulang.
