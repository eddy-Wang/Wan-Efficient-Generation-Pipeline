| config | status | n | total_s | realtime_factor | peak_gib | warmup_s | first_frame_s | steady_chunk_s | psnr_vs_full_vae_db |
|---|---|---|---|---|---|---|---|---|---|
| A  bf16, whole-clip decode | ok | 4 | 12.06 | 0.42 | 23.48 | 11.96 | nan | nan | nan |
| B  fp16, streaming decode | ok | 4 | 12.71 | 0.40 | 25.13 | 14.26 | 1.84 | 1.80 | 36.97 |
| C  fp16, streaming, light VAE | ok | 4 | 7.36 | 0.69 | 25.00 | 7.31 | 0.80 | 1.08 | 28.40 |
| D  B + torch.compile | ok | 4 | 10.45 | 0.48 | 27.76 | 255.30 | 1.53 | 1.48 | 36.68 |
| E  C + torch.compile | ok | 4 | 6.69 | 0.76 | 25.00 | 6.64 | 0.69 | 0.98 | 28.25 |
