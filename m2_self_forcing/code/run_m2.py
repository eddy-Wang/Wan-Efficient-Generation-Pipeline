"""Batch runner for M2 (Self Forcing): one process, many prompts, many configs.

Run from the Self-Forcing repo root:
  python ../m2/run_m2.py --prompts ../m2/prompts_expanded_v1.jsonl --out ../outputs --suites main,ablate,drift
Every video is skipped if its mp4 and .json record already exist, so re-submitting continues the work.
"""
import argparse, contextlib, hashlib, io, json, os, random, re, sys, time

sys.path.insert(0, os.getcwd())
import imageio
import torch
from einops import rearrange
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline

SEED = 20261027
FPS = 16
BASE = dict(steps=[1000, 750, 500, 250], L=-1, sink=0, chunk=3, frames=21, ctx=0, seed=SEED)


def cfg(name, subset, **kw):
    c = dict(BASE)
    c.update(kw)
    c.update(name=name, subset=subset)
    return c


SUITES = {
    "smoke": [cfg("smoke_main", "smoke2"), cfg("smoke_long", "smoke1", L=21, frames=81)],
    "main": [cfg("m2_main", "all")],
    "ablate": [
        cfg("m2_steps1", "sub30", steps=[1000]),
        cfg("m2_steps2", "sub30", steps=[1000, 500]),
        cfg("m2_steps3", "sub30", steps=[1000, 750, 500]),
        cfg("m2_win6", "sub30", L=6),
        cfg("m2_win9", "sub30", L=9),
        cfg("m2_win12", "sub30", L=12),
        cfg("m2_win15", "sub30", L=15),
        cfg("m2_win9_sink3", "sub30", L=9, sink=3),
        cfg("m2_chunk7", "sub30", chunk=7),
        cfg("m2_chunk1", "sub30", chunk=1),
    ],
    "drift": [
        cfg("m2_long_w21", "drift20", L=21, frames=81),
        cfg("m2_long_w21_sink3", "drift20", L=21, sink=3, frames=81),
        cfg("m2_long_w12", "drift20", L=12, frames=81),
        cfg("m2_long_w12_sink3", "drift20", L=12, sink=3, frames=81),
        cfg("m2_long_w21_ctx50", "drift20", L=21, frames=81, ctx=50),
        cfg("m2_long_w21_sink3_ctx50", "drift20", L=21, sink=3, frames=81, ctx=50),
        cfg("m2_long_w21_sink3_ctx25", "drift20", L=21, sink=3, frames=81, ctx=25),
        cfg("m2_long_w21_sink3_ctx100", "drift20", L=21, sink=3, frames=81, ctx=100),
    ],
}

# more prompts for the long rollouts (same configs and directories, ten new prompts)
SUITES["drift_more"] = [dict(c, subset="drift_extra10") for c in SUITES["drift"]]
# two more seeds on ten prompts: default 5 s setting and the three key long-rollout settings
SUITES["seeds"] = []
for _k, _sd in ((2, SEED + 1), (3, SEED + 2)):
    SUITES["seeds"] += [
        cfg(f"m2_seed{_k}_main", "seed10", seed=_sd),
        cfg(f"m2_seed{_k}_long_w21", "seed10", L=21, frames=81, seed=_sd),
        cfg(f"m2_seed{_k}_long_w21_sink3", "seed10", L=21, sink=3, frames=81, seed=_sd),
        cfg(f"m2_seed{_k}_long_w12_sink3", "seed10", L=12, sink=3, frames=81, seed=_sd),
    ]


def pick_subsets(rows):
    ids = [r["prompt_id"] for r in rows]
    by_src = {}
    for r in rows:
        by_src.setdefault(r.get("source", ""), []).append(r["prompt_id"])
    rng = random.Random(1234)
    sub30 = []
    srcs = sorted(by_src)
    # 20 official + 10 self-authored, stable across runs
    for s in srcs:
        k = 10 if "self" in s else 20
        sub30 += rng.sample(sorted(by_src[s]), k)
    sub30 = sorted(sub30)
    shuf = list(sub30)
    random.Random(99).shuffle(shuf)
    return {"all": ids, "sub30": sub30, "drift20": sorted(shuf[:20]), "drift_extra10": sorted(shuf[20:30]),
            "seed10": sorted(shuf[:10]), "smoke2": ids[:2], "smoke1": ids[:1]}


class GenProxy(torch.nn.Module):
    """Wraps the generator so the KV-cache refresh pass can use a re-noised context (improvement knob)."""

    def __init__(self, gen, scheduler):
        super().__init__()
        self.gen = gen
        self.sched = scheduler
        self.ctx = 0

    def forward(self, *a, **kw):
        t = kw.get("timestep")
        if self.ctx > 0 and kw.get("kv_cache") is not None and t is not None and bool(torch.all(t == self.ctx)):
            x = kw["noisy_image_or_video"]
            flat = x.flatten(0, 1)
            kw["noisy_image_or_video"] = self.sched.add_noise(
                flat, torch.randn_like(flat), t.flatten()).unflatten(0, x.shape[:2])
        return self.gen(*a, **kw)


def apply_config(pipe, proxy, c):
    sched_ts = torch.cat((pipe.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    pipe.denoising_step_list = sched_ts[1000 - torch.tensor(c["steps"], dtype=torch.long)]
    model = proxy.gen.model
    model.local_attn_size = c["L"]
    for blk in model.blocks:
        sa = blk.self_attn
        sa.local_attn_size = c["L"]
        sa.sink_size = c["sink"]
        sa.max_attention_size = 32760 if c["L"] == -1 else c["L"] * 1560
    pipe.local_attn_size = c["L"]
    pipe.num_frame_per_block = c["chunk"]
    if c["chunk"] > 1:
        model.num_frame_per_block = c["chunk"]
    pipe.kv_cache1 = None  # re-allocated with the right size on the next call
    pipe.args.context_noise = c["ctx"]
    proxy.ctx = c["ctx"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def one_video(pipe, c, prompt, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    noise = torch.randn([1, c["frames"], 16, 60, 104], device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        video = pipe.inference(noise=noise, text_prompts=[prompt], return_latents=False, profile=True)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    log = buf.getvalue()
    num = lambda pat: [float(x) for x in re.findall(pat, log)]
    perf = {
        "success": True,
        "generation_seconds": secs,
        "scope": "pipeline.inference: text encoding + KV init + denoising + VAE decode; excludes model load and mp4 export",
        "init_ms": (num(r"Initialization/caching time: ([\d.]+) ms") or [None])[0],
        "diffusion_ms": (num(r"Diffusion generation time: ([\d.]+) ms") or [None])[0],
        "vae_ms": (num(r"VAE decoding time: ([\d.]+) ms") or [None])[0],
        "block_ms": num(r"Block \d+ generation time: ([\d.]+) ms"),
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_scope": "torch allocator peak inside pipeline.inference (reset before each video)",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }
    v = rearrange(video, "b t c h w -> b t h w c")[0].float().cpu()
    return (v * 255).round().clamp(0, 255).to(torch.uint8), perf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="smoke")
    ap.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--budget-minutes", type=float, default=330)
    args = ap.parse_args()
    deadline = time.monotonic() + args.budget_minutes * 60

    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8")]
    prompt_of = {r["prompt_id"]: r["prompt"] for r in rows}
    subsets = pick_subsets(rows)
    os.makedirs(args.out, exist_ok=True)
    json.dump(subsets, open(os.path.join(args.out, "subsets.json"), "w"), indent=1)

    torch.set_grad_enabled(False)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    pipe = CausalInferencePipeline(config, device=torch.device("cuda"))
    sd = torch.load(args.checkpoint_path, map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema" if args.use_ema else "generator"])
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device="cuda")
    pipe.generator.to(device="cuda")
    pipe.vae.to(device="cuda")
    proxy = GenProxy(pipe.generator, pipe.scheduler)
    pipe.generator = proxy
    print("model ready", flush=True)

    # warmup (not saved, not timed): removes CUDA init and kernel-selection cost from the first real video
    apply_config(pipe, proxy, BASE)
    one_video(pipe, BASE, prompt_of[rows[0]["prompt_id"]], SEED)
    print("warmup done", flush=True)

    n_done = n_skip = 0
    for suite in args.suites.split(","):
        for c in SUITES[suite]:
            apply_config(pipe, proxy, c)
            d = os.path.join(args.out, c["name"])
            os.makedirs(d, exist_ok=True)
            for pid in subsets[c["subset"]]:
                seed = c["seed"]
                mp4, rec = os.path.join(d, f"{pid}_{seed}.mp4"), os.path.join(d, f"{pid}_{seed}.json")
                if os.path.exists(mp4) and os.path.exists(rec):
                    n_skip += 1
                    continue
                if time.monotonic() > deadline:
                    print("time budget reached, stopping; re-submit to continue", flush=True)
                    print(f"done={n_done} skipped={n_skip}")
                    return
                video, perf = one_video(pipe, c, prompt_of[pid], seed)
                tmp = mp4 + ".partial.mp4"
                # same encoder settings as Wan's own save_video, so bitrate does not differ between methods
                writer = imageio.get_writer(tmp, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
                for frame in video.numpy():
                    writer.append_data(frame)
                writer.close()
                os.replace(tmp, mp4)
                record = {"schema": "genai-m2-v1", "method": "self_forcing", "config_name": c["name"],
                          "prompt_id": pid, "seed": seed, "config": c,
                          "video": {"frames": int(video.shape[0]), "height": int(video.shape[1]),
                                    "width": int(video.shape[2]), "fps": FPS, "sha256": sha256(mp4)},
                          "performance": perf, "checkpoint": os.path.basename(args.checkpoint_path),
                          "use_ema": bool(args.use_ema),
                          "encoder": "imageio libx264 quality=8 macro_block_size=1 (as Wan save_video)"}
                json.dump(record, open(rec + ".tmp", "w"))
                os.replace(rec + ".tmp", rec)
                n_done += 1
                print(f"[{c['name']}] {pid} {perf['generation_seconds']:.1f}s "
                      f"peak={perf['peak_torch_allocated_bytes'] / 2**30:.1f}GiB frames={video.shape[0]}", flush=True)
    print(f"ALL DONE done={n_done} skipped={n_skip}", flush=True)


if __name__ == "__main__":
    main()
