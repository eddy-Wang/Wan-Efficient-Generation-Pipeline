"""Streaming-speed benchmark for Self Forcing (M2): what does it take to reach real time on one L40S?

Replicates the authors' demo loop (chunk-wise generation, per-chunk decoding) without the web server, and times:
text encoding, cache init, per-chunk denoising, cache refresh and decoding. Configurations are run one after another
in one process (the order matters: torch.compile is applied in place and stays on).

  A  bf16, whole-clip decode at the end (the setting of all other M2 experiments)
  B  fp16, per-chunk streaming decode with the block VAE decoder
  C  fp16, per-chunk streaming decode with the light TAEHV decoder
  D  B + torch.compile (transformer and decoder)
  E  C + torch.compile (transformer)

Run from the Self-Forcing repo root:  python ../m2/bench_stream.py --out ../bench/bench_stream.json
"""
import argparse, json, os, sys, time, traceback

sys.path.insert(0, os.getcwd())
import torch
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline

SEED = 20261027
FPS = 16
NUM_BLOCKS = 7
CHUNK = 3
VIDEO_SECONDS = 81 / FPS


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def load_prompts(path, n):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    sub30 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(path)), "..", "outputs", "subsets.json")))["sub30"]
    by = {r["prompt_id"]: r["prompt"] for r in rows}
    return [(pid, by[pid]) for pid in sub30[:n]]


def build_decoders(device):
    from demo_utils.vae_block3 import VAEDecoderWrapper
    block = VAEDecoderWrapper()
    sd = torch.load("wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", map_location="cpu")
    block.load_state_dict({k: v for k, v in sd.items() if "decoder." in k or "conv2" in k})
    block.eval().to(dtype=torch.float16).requires_grad_(False).to(device)
    taehv = None
    ck = "checkpoints/taew2_1.pth"
    if os.path.exists(ck):
        from demo_utils.taehv import TAEHV
        taehv = TAEHV(checkpoint_path=ck).to(torch.float16).eval().requires_grad_(False).to(device)
    return block, taehv


def stream_one(pipe, prompt, decoder_kind, block_dec, taehv, dtype, steps, check_full=False):
    """One video, generated chunk by chunk. Returns timings (s) and, optionally, pixels for a quality check."""
    from demo_utils.constant import ZERO_VAE_CACHE
    dev = torch.device("cuda")
    t_start = sync()
    cond = pipe.text_encoder(text_prompts=[prompt])
    for k in cond:
        cond[k] = cond[k].to(dtype=dtype)
    t_text = sync() - t_start

    t0 = sync()
    pipe._initialize_kv_cache(batch_size=1, dtype=dtype, device=dev)
    pipe._initialize_crossattn_cache(batch_size=1, dtype=dtype, device=dev)
    gen = torch.Generator(dev).manual_seed(SEED)
    noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=dtype, generator=gen)
    vae_cache = None
    if decoder_kind == "block":
        vae_cache = [c.to(device=dev, dtype=torch.float16) for c in ZERO_VAE_CACHE]
    t_init = sync() - t0

    rows, frames, latents = [], [], []
    ts = pipe.denoising_step_list
    for idx in range(NUM_BLOCKS):
        s = idx * CHUNK
        noisy = noise[:, s:s + CHUNK]
        tb = sync()
        for j, t in enumerate(ts):
            timestep = torch.ones([1, CHUNK], device=dev, dtype=torch.int64) * t
            _, pred = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=timestep,
                                     kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                                     current_start=s * pipe.frame_seq_length)
            if j < len(ts) - 1:
                noisy = pipe.scheduler.add_noise(
                    pred.flatten(0, 1), torch.randn_like(pred.flatten(0, 1)),
                    ts[j + 1] * torch.ones([CHUNK], device=dev, dtype=torch.long)).unflatten(0, pred.shape[:2])
        t_den = sync() - tb
        t_ref = 0.0
        if idx != NUM_BLOCKS - 1:  # as in the authors' demo, the last chunk needs no cache refresh
            tr = sync()
            pipe.generator(noisy_image_or_video=pred, conditional_dict=cond, timestep=torch.zeros_like(timestep),
                           kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                           current_start=s * pipe.frame_seq_length)
            t_ref = sync() - tr
        td = sync()
        if decoder_kind == "block":
            pix, vae_cache = block_dec(pred.half(), *vae_cache)
            if idx == 0:
                pix = pix[:, 3:]
        else:
            ctx = pred if idx == 0 else torch.cat([prev, pred], dim=1)
            prev = pred[:, -3:] if idx > 0 else pred
            pix = taehv.decode_video(ctx, parallel=False).mul_(2).sub_(1)
            pix = pix[:, 3:] if idx == 0 else pix[:, 12:]
        t_dec = sync() - td
        rows.append(dict(block=idx, denoise_s=t_den, refresh_s=t_ref, decode_s=t_dec, total_s=t_den + t_ref + t_dec))
        if check_full:
            latents.append(pred.detach().clone())
            frames.append(pix.detach().float().clamp(-1, 1).cpu())
    total = sync() - t_start
    out = dict(text_s=t_text, init_s=t_init, blocks=rows, total_s=total,
               first_frame_s=t_text + t_init + rows[0]["total_s"],
               steady_block_s=sum(r["total_s"] for r in rows[1:]) / (NUM_BLOCKS - 1),
               peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    if check_full:
        out["_latents"] = torch.cat(latents, dim=1)
        out["_frames"] = torch.cat(frames, dim=1)
    return out


def psnr_vs_full(pipe, res):
    """PSNR (dB) of streamed pixels against the standard whole-clip VAE decode of the same latents."""
    with torch.no_grad():
        # the pipeline maps this output with (x * 0.5 + 0.5), so it is in [-1, 1], like the streamed pixels
        ref = pipe.vae.decode_to_pixel(res["_latents"].to(torch.bfloat16), use_cache=False).float().clamp(-1, 1).cpu()
    a, b = res["_frames"], ref
    n = min(a.shape[1], b.shape[1])
    mse = ((a[:, :n] - b[:, :n]) ** 2).mean().item() / 4.0  # rescale from [-1,1] to [0,1]
    return float(10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12)))), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="../m2/prompts_expanded_v1.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4, help="timed videos per configuration")
    ap.add_argument("--configs", default="A,B,C,D,E")
    ap.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    pipe = CausalInferencePipeline(config, device=dev)
    sd = torch.load(args.checkpoint_path, map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema"])
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device=dev)
    pipe.generator.to(device=dev)
    pipe.vae.to(device=dev)
    block_dec, taehv = build_decoders(dev)
    prompts = load_prompts(args.prompts, args.n + 1)
    warm, timed = prompts[0], prompts[1:]
    results = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, taehv_available=taehv is not None,
                   video_seconds=VIDEO_SECONDS, configs={})
    if os.path.exists(args.out):  # keep configurations measured in an earlier run; the ones requested now replace them
        results["configs"] = json.load(open(args.out)).get("configs", {})

    def save():
        json.dump(results, open(args.out, "w"), indent=1)

    plan = {"A": dict(dtype=torch.bfloat16, dec="whole", compile=False),
            "B": dict(dtype=torch.float16, dec="block", compile=False),
            "C": dict(dtype=torch.float16, dec="taehv", compile=False),
            "D": dict(dtype=torch.float16, dec="block", compile=True),
            "E": dict(dtype=torch.float16, dec="taehv", compile=True)}
    compiled = False
    for name in args.configs.split(","):
        p = plan[name]
        try:
            if p["dec"] == "taehv" and taehv is None:
                raise RuntimeError("taew2_1.pth not found in checkpoints/")
            if p["dtype"] == torch.float16:
                pipe.generator.to(torch.float16)
                pipe.text_encoder.to(torch.bfloat16)
            if p["compile"] and not compiled:
                t = time.perf_counter()
                pipe.generator.model.compile(mode="max-autotune-no-cudagraphs")
                if p["dec"] == "block":
                    block_dec.compile(mode="max-autotune-no-cudagraphs")
                compiled = True
                print(f"[{name}] compile requested ({time.perf_counter() - t:.1f}s to set up; compiles on first call)", flush=True)
            entry = dict(**{k: (str(v) if k == "dtype" else v) for k, v in p.items()}, videos=[])
            # one warm-up video (this is where torch.compile pays its cost); timed separately
            tw = time.perf_counter()
            if p["dec"] == "whole":
                with torch.no_grad():
                    noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
                    pipe.inference(noise=noise, text_prompts=[warm[1]])
            else:
                stream_one(pipe, warm[1], p["dec"], block_dec, taehv, p["dtype"], len(pipe.denoising_step_list))
            entry["warmup_seconds"] = time.perf_counter() - tw
            print(f"[{name}] warm-up {entry['warmup_seconds']:.1f}s", flush=True)
            for i, (pid, prompt) in enumerate(timed):
                torch.cuda.reset_peak_memory_stats()
                if p["dec"] == "whole":
                    noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16,
                                        generator=torch.Generator(dev).manual_seed(SEED))
                    t = sync()
                    pipe.inference(noise=noise, text_prompts=[prompt])
                    r = dict(total_s=sync() - t, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
                else:
                    r = stream_one(pipe, prompt, p["dec"], block_dec, taehv, p["dtype"], 4,
                                   check_full=(p["dec"] == "taehv" or (p["dec"] == "block" and i < 2)))
                    if "_latents" in r:
                        r["psnr_vs_full_vae_db"], r["psnr_frames"] = psnr_vs_full(pipe, r)
                        del r["_latents"], r["_frames"]
                r["prompt_id"] = pid
                r["realtime_factor"] = VIDEO_SECONDS / r["total_s"]
                entry["videos"].append(r)
                print(f"[{name}] {pid} total={r['total_s']:.2f}s rtf={r['realtime_factor']:.2f}"
                      + (f" first_frame={r['first_frame_s']:.2f}s steady_block={r['steady_block_s']:.2f}s"
                         if "first_frame_s" in r else ""), flush=True)
            results["configs"][name] = entry
        except Exception as e:  # keep going: one failed configuration should not lose the others
            results["configs"][name] = dict(error=repr(e), trace=traceback.format_exc()[-1500:])
            print(f"[{name}] FAILED: {e!r}", flush=True)
        save()
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
