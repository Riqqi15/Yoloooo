# Batch review LabelMe — 2026-09-08

Batch ini berisi 20 foto stasiun dari Wikimedia Commons dengan lisensi yang tercatat di `sources.csv`. JSON di folder ini adalah polygon awal yang dibuat otomatis dari pemeriksaan visual; statusnya **provisional**, belum boleh masuk training.

## Cara merevisi

1. Buka LabelMe → **Open Dir** dan pilih folder batch ini.
2. Buka tiap foto, pilih polygon `tactile_paving`, lalu geser/tambah/hapus titik agar hanya menutup permukaan tactile yang terlihat.
3. Jika foto ternyata bukan konteks stasiun atau tactile tidak dapat dipastikan, hapus seluruh shape dan beri tahu saya untuk memasukkannya sebagai hard negative/mengecualikannya.
4. Simpan (`Ctrl+S`) sehingga JSON di folder ini diperbarui.

Prioritas pengecekan: garis tepi peron jangan ikut terlabel, jangan menutup rel/lantai biasa, dan pastikan seluruh cabang/strip tactile yang jelas ikut masuk. `sources.csv` jangan dihapus karena dipakai untuk atribusi dan audit lisensi.
