# Training Label Gate Design

## Goal

Mencegah anotasi salah, provisional, ambiguous, berubah setelah review, atau berasal dari test set masuk export YOLO untuk training.

## Scope

Satu tool menangani validasi anotasi, render overlay QA, pencatatan keputusan reviewer, dan export YOLO. Input hanya manifest intake stasiun dari `build_training_manifest.py`. Tool tidak melakukan auto-label, training, atau menganggap pemeriksaan format sebagai kebenaran visual.

## Data contract

Setiap gambar memiliki JSON anotasi bernama `<sample_id>.json` berisi `schema_version`, `source_sha256`, daftar `objects` dengan box `xyxy`, dan daftar `tactile` dengan polygon. Label harus ada di taxonomy versioned.

Persetujuan disimpan terpisah di `review_manifest.csv`. Row mengikat `sample_id`, hash sumber, dan hash file anotasi. Perubahan satu byte pada sumber atau anotasi membatalkan review. Status yang didukung: `human_reviewed` dan `ambiguous`.

## Commands

- `render`: validasi struktur/geometri lalu buat overlay sumber-resolusi untuk review.
- `review`: setelah manusia memeriksa overlay, ikat hash anotasi aktif ke identitas reviewer dan timestamp.
- `ambiguous`: catat sampel yang tidak dapat diputuskan; sampel tetap gagal export.
- `validate`: keluarkan laporan error, distribusi kelas/split/kondisi, dan status `training_ready`.
- `export`: hanya berjalan bila taxonomy `approved`, semua sampel valid dan `human_reviewed`, tidak ada hash test, dan review masih cocok. Output object/tactile YOLO dibuat melalui staging lalu commit atomik.

## Safety gates

- Konteks harus `station_environment`.
- Hash sumber harus cocok dengan intake dan tidak sama dengan 10 foto `test_only`.
- Box finite, punya luas, dan berada dalam gambar.
- Polygon finite, minimal tiga titik, punya luas, dan berada dalam gambar.
- Task yang tidak diminta harus kosong.
- Label asing ditolak.
- Empty annotation boleh sebagai negative sample, tetapi tetap membutuhkan review manusia.
- Warna kuning tidak dipakai sebagai validator otomatis karena tactile tidak selalu kuning dan cat kuning dapat menjadi hard negative.

## Testing

Tes mencakup provisional/unreviewed, review hash stale, label asing, geometri di luar gambar, protected test hash, overlay, normalisasi YOLO, dan export atomik. Workspace bukan Git repository, jadi tidak ada commit/worktree pada implementasi ini.
