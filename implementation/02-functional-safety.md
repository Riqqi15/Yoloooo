# Functional Safety

## Posisi keselamatan

Fitur ini bukan alat mobilitas tersertifikasi dan tidak dapat menjamin tidak terjadi cedera. Deployment mandiri di stasiun nyata berada di luar cakupan demo. Penggunaan awal selalu di lingkungan terkontrol bersama pengawas yang dapat melihat.

Prinsip inti:

```text
Tidak terdeteksi != aman
Satu sinyal bahaya kredibel -> STOP
Ragu/error/stale -> STOP
Lepas STOP hanya setelah seluruh pemeriksaan stabil
```

## Bahasa yang dilarang

Jangan mengucapkan:

- “Jalur aman.”
- “Tidak ada bahaya.”
- “Silakan maju dengan aman.”

Gunakan:

- “Jalur terdeteksi di depan.”
- “Tidak ada bahaya yang terdeteksi saat ini. Tetap gunakan alat bantu.”
- “Kondisi belum dapat dipastikan. Berhenti.”

## Watchdog independen

Health monitor tidak bergantung pada hasil YOLO. Ia memeriksa:

- Frame kamera masih berubah.
- Timestamp frame masih baru.
- Latency tidak melewati batas berulang.
- Model tetap loaded.
- Depth status valid ketika dipakai.
- Kamera menghadap arah masuk akal.
- Aplikasi aktif di foreground.
- TTS dan haptic dapat dipanggil.
- Tekanan RAM, thermal throttling, serta battery critical.

Jika tidak ada keputusan segar sekitar 500 ms, state menjadi `systemFailure` dan STOP dikirim.

## Kegagalan internal

| Kegagalan | Deteksi | Respons |
|---|---|---|
| Kamera freeze | frame hash/timestamp tidak berubah | batalkan arahan, STOP |
| Permission dicabut | camera exception | STOP dan jelaskan permission |
| Model gagal load | exception/checksum mismatch | jangan mulai panduan |
| Class metadata salah | manifest validation | blokir model |
| Inference lambat | rolling p95/timeout | degraded atau STOP |
| Hasil stale | `capturedAt` melewati TTL | buang hasil |
| TTS macet | completion/error callback | bunyi/getaran STOP |
| Antrean suara basi | decision version berubah | flush queue |
| RAM pressure | allocation failure/OS signal | stop stream, release model, STOP |
| Perangkat panas | performance/thermal signal | turunkan profil atau STOP |
| Aplikasi background | lifecycle event | hentikan kamera dan seluruh panduan |

“Memori” pada tabel berarti RAM runtime, bukan storage Neon. Camera buffers, tensors, model, dan depth maps memakai RAM. Backpressure dan reuse buffer wajib mencegah frame menumpuk.

## Kegagalan eksternal

| Kondisi | Deteksi | Respons |
|---|---|---|
| Gelap/silau | luminance dan contrast gate | STOP, minta perbaiki cahaya |
| Lens tertutup/kotor | blur, area gelap, frame statis | STOP, minta periksa kamera |
| HP terlalu miring | orientation sensor | STOP, minta koreksi posisi |
| Motion blur | blur/optical quality score | minta berjalan lebih pelan atau STOP |
| Kerumunan menutup jalur | mask hilang + banyak objek | STOP |
| Lantai reflektif/basah | depth/vision tidak stabil | uncertain + STOP |
| GPS buruk | accuracy di luar batas | jangan tampilkan `You Are Here` |
| Kebisingan | tidak dapat dipastikan via kamera | gunakan getaran dan open-ear audio |

## Kaca dan objek transparan

Mitigasi berlapis:

1. Fine-tune kelas `glass_door` menggunakan bingkai, gagang, sambungan, stiker, dan refleksi.
2. Periksa depth sebagai sumber kedua.
3. Periksa floor continuity, garis vertikal, dan perubahan permukaan.
4. Jika petunjuk vision/depth lemah, hasil `uncertain` dan STOP.
5. Pengembangan berikutnya dapat menambahkan ultrasonic atau ToF wearable.

Kaca sangat bersih dan kabel tipis tetap memiliki residual risk tinggi. RGB camera, ARCore, ToF, dan ultrasonic masing-masing dapat melewatkan kondisi tertentu. Tongkat/pendamping tetap lapisan keselamatan fisik.

## Lubang dan perubahan tinggi

- YOLO memberi dugaan kelas `hole`, `step_up`, `step_down`, `stairs`, atau `platform_edge`.
- Depth mencari discontinuity lantai.
- Salah satu sinyal bahaya yang kredibel cukup untuk STOP.
- Tidak boleh meminta kedua sumber sepakat untuk menghentikan pengguna karena ini dapat meningkatkan false negative.
- Untuk melepas STOP, sumber yang relevan harus kembali stabil.

## Audio safety

- STOP memakai bunyi pendek yang unik dan pola getaran panjang.
- STOP memotong seluruh arahan lama.
- Pesan arah tidak boleh tertunda di belakang deskripsi objek.
- Headphone yang menutup suara lingkungan tidak disarankan. Project Guideline merekomendasikan open-ear audio agar suara sekitar tetap terdengar.
- Emergency stop harus dapat digunakan dengan TalkBack atau perintah suara.

## Security dan privasi sebagai safety

- Verifikasi SHA-256 model sebelum load.
- APK/update ditandatangani.
- Backend opsional memakai HTTPS dan authentication.
- Kamera diproses lokal secara default.
- Tidak menyimpan gambar/video tanpa persetujuan eksplisit.
- Log tidak mengandung wajah, gambar, token, atau lokasi presisi secara default.
- Model dan threshold memiliki versi sehingga incident dapat direproduksi.

## Risk register awal

| Risiko | Penyebab | Dampak | Mitigasi | Residual risk |
|---|---|---|---|---|
| Lubang terlewat | blur/domain shift | jatuh | YOLO + depth + quality gate | tinggi sampai diuji luas |
| Kamera freeze | runtime/driver | arahan basi | watchdog + TTL | rendah setelah mitigasi |
| Jalur palsu | lantai kuning/pola | arah salah | negative data + multi-frame | sedang |
| Jalur tertutup dianggap lanjut | occlusion | tabrakan | intersection + STOP | sedang |
| Kaca tidak terlihat | transparansi | tabrakan | custom label + depth + tongkat | tinggi |
| Kabel tipis terlewat | resolusi rendah | tersandung | alat bantu fisik + catat batas | tinggi |
| Depth salah | unsupported/low texture | jarak salah | runtime check + distance band | sedang |
| HP terlalu lambat | CPU/GPU lama | warning terlambat | startup benchmark + disable unsafe mode | rendah setelah fail-safe |
| TTS terlambat | queue/audio route | warning terlambat | dedicated STOP sound + haptic | sedang |
| GPS salah | indoor signal | node salah | accuracy/radius threshold | rendah |

## Pengujian aman

Urutan:

1. Unit test decision engine.
2. Replay gambar/video.
3. Kamera statis.
4. Operator yang dapat melihat.
5. Lingkungan terkontrol dengan benda lunak.
6. Review ahli Orientation and Mobility jika tersedia.
7. Pengguna tunanetra hanya dengan pendamping profesional dan prosedur etik yang sesuai.

Larangan:

- Tidak membuat lubang atau tepi nyata untuk demo.
- Tidak menguji dekat rel, jalan, tangga, atau tepi peron.
- Tidak meminta orang menutup mata untuk simulasi bahaya.
- Tidak menguji pengguna tunanetra sendirian.
- Tidak mengklaim hasil meeting room berlaku otomatis di stasiun.

## Gate keselamatan

Panduan tidak boleh aktif jika salah satu kondisi berikut terjadi:

- Camera health gagal.
- Model manifest/checksum gagal.
- Startup benchmark melewati batas profil.
- Output STOP tidak dapat berbunyi atau bergetar.
- Device orientation tidak valid.
- Hasil inference stale.
- Capability profile belum dipilih.

Safety gate tidak boleh dapat dilewati oleh UI demo biasa.
