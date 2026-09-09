# Batch review LabelMe — 2026-09-08

Batch ini awalnya berisi 20 foto stasiun dari Wikimedia Commons dengan lisensi yang tercatat di `sources.csv`. Setelah revisi manual, 19 foto memiliki anotasi `tactile_paving` yang diterima untuk tahap validasi berikutnya.

Foto `commons_19822898_Paris_Est_caddie_sur_un_quai.jpg` dikecualikan karena menampilkan troli di peron dan tidak dipakai untuk training. Berkas sumbernya disimpan di folder `excluded/` untuk audit lisensi, tanpa JSON anotasi dan tanpa baris di `intake.csv`.

## Cara merevisi

1. Buka LabelMe → **Open Dir** dan pilih folder batch ini.
2. Buka tiap foto, pilih polygon `tactile_paving`, lalu geser/tambah/hapus titik agar hanya menutup permukaan tactile yang terlihat.
3. Jika foto ternyata bukan konteks stasiun atau tactile tidak dapat dipastikan, hapus seluruh shape dan beri tahu saya untuk memasukkannya sebagai hard negative/mengecualikannya.
4. Simpan (`Ctrl+S`) sehingga JSON di folder ini diperbarui.

Prioritas pengecekan: garis tepi peron jangan ikut terlabel, jangan menutup rel/lantai biasa, dan pastikan seluruh cabang/strip tactile yang jelas ikut masuk. `sources.csv` jangan dihapus karena dipakai untuk atribusi dan audit lisensi.
