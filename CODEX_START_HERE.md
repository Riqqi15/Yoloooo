# Codex Start Here — AI Camera Tactile Model

**Wajib dibaca oleh Codex sebelum melakukan pekerjaan apa pun di repository ini.**

Dokumen ini adalah manifest onboarding. `AGENTS.md` menunjuk ke sini agar Codex yang membuka atau meng-clone repository mendapat pemberitahuan tentang konteks yang harus dibaca sebelum mengubah kode, data, notebook, model, evaluasi, atau integrasi mobile.

## Prosedur wajib sebelum bekerja

1. Baca [`AGENTS.md`](AGENTS.md) dan ikuti aturan repository.
2. Baca seluruh [`docs/HANDOFF_TACTILE_V3.md`](docs/HANDOFF_TACTILE_V3.md). Ini adalah sumber konteks status, score sementara, target produk, artefak, dan daftar pekerjaan teman berikutnya.
3. Baca [`README.md`](README.md) untuk scope dataset, kontrak model, batas keselamatan, dan urutan pengerjaan.
4. Untuk pekerjaan Colab/training, baca juga:
   - [`docs/superpowers/plans/2026-09-10-colab-training-handoff.md`](docs/superpowers/plans/2026-09-10-colab-training-handoff.md)
   - [`docs/superpowers/specs/2026-09-10-colab-training-handoff-design.md`](docs/superpowers/specs/2026-09-10-colab-training-handoff-design.md)
5. Sebelum menyentuh file, nyatakan secara internal bahwa dokumen yang relevan sudah dibaca dan cek status Git agar perubahan pengguna tidak tertimpa.

## Fakta yang tidak boleh diabaikan

- Kandidat saat ini hanya **segmentasi satu kelas `tactile_paving`**.
- Kandidat **belum mobile-ready** dan bukan sistem keselamatan navigasi lengkap.
- Model belum mendeteksi orang, penghalang, lubang, rel, tepi peron, arah belok, atau jarak 5 meter.
- Hazard, frame stale, confidence rendah, atau kondisi ambigu harus berujung `STOP`/`Uncertain`; sistem tidak boleh menebak aman.
- Jangan melatih dengan protected ground truth, jangan mengembalikan sumber troli yang sudah dikecualikan, dan jangan menghapus provenance, manifest, atau checksum.
- Artefak besar wajib memakai Git LFS.

## Peta dokumen

| Kebutuhan | Dokumen utama |
| --- | --- |
| Status, score, target, dan handoff | [`docs/HANDOFF_TACTILE_V3.md`](docs/HANDOFF_TACTILE_V3.md) |
| Scope umum dan aturan keselamatan | [`README.md`](README.md) |
| Training Colab yang reproducible | [`docs/superpowers/plans/2026-09-10-colab-training-handoff.md`](docs/superpowers/plans/2026-09-10-colab-training-handoff.md) |
| Alasan desain dan gate | [`docs/superpowers/specs/2026-09-10-colab-training-handoff-design.md`](docs/superpowers/specs/2026-09-10-colab-training-handoff-design.md) |
| Notebook yang dibagikan | [`notebooks/train_tactile_v3_public_colab.ipynb`](notebooks/train_tactile_v3_public_colab.ipynb) |

## Aturan pemeliharaan

Jika status model, score, branch, notebook, dataset, target, atau langkah handoff berubah, perbarui `docs/HANDOFF_TACTILE_V3.md` dan dokumen rencana yang terkait pada perubahan yang sama. Jika aturan onboarding berubah, perbarui file ini dan `AGENTS.md` bersama-sama.
