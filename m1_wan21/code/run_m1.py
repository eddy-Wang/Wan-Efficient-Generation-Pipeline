"""Batch runner for M1 (Wan2.1-T2V-1.3B, the teacher of Self Forcing), measured exactly like M2.

One process, model loaded once, every video seeded separately, finished videos skipped (re-submit to continue).
generation_seconds = one WanT2V.generate call (text encoding + denoising + VAE decoding), excluding model load and export.
Videos are written with Wan's own cache_video (imageio libx264 quality=8), the same encoder settings as M2.

  python run_m1.py --wan_repo ~/sfproj/Wan2.1 --ckpt ~/sfproj/Self-Forcing/wan_models/Wan2.1-T2V-1.3B \
      --prompts ~/sfproj/m2/prompts_expanded_v1.jsonl --subsets ~/sfproj/outputs/subsets.json --out ~/sfproj/outputs_m1 \
      --suites smoke,main,steps
"""
import argparse, hashlib, json, os, sys, time

SEED = 20261027
FPS = 16
BASE = dict(steps=50, cfg=6.0, shift=8.0, solver="unipc", frames=81, width=832, height=480, offload=False, seed=SEED)


def cfg(name, subset, **kw):
    c = dict(BASE)
    c.update(kw)
    c.update(name=name, subset=subset)
    return c


SUITES = {
    "smoke": [cfg("m1_smoke_4steps", "one", steps=4)],
    "main": [cfg("m1_main", "all")],
    # the teacher with the student's budget: is distillation needed, or would fewer teacher steps do?
    "steps": [cfg("m1_steps4", "sub30", steps=4), cfg("m1_steps8", "sub30", steps=8),
              cfg("m1_steps16", "sub30", steps=16)],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan_repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--subsets", required=True, help="subsets.json written by the M2 runner (same 30-prompt subset)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="smoke,main,steps")
    ap.add_argument("--budget-minutes", type=float, default=330)
    a = ap.parse_args()
    deadline = time.monotonic() + a.budget_minutes * 60
    sys.path.insert(0, os.path.expanduser(a.wan_repo))
    import torch
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video

    rows = [json.loads(l) for l in open(os.path.expanduser(a.prompts), encoding="utf-8")]
    prompt_of = {r["prompt_id"]: r["prompt"] for r in rows}
    sub = json.load(open(os.path.expanduser(a.subsets)))
    subsets = {"all": [r["prompt_id"] for r in rows], "sub30": sub["sub30"], "one": [rows[0]["prompt_id"]]}
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    t_load = time.perf_counter()
    pipe = wan.WanT2V(config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=os.path.expanduser(a.ckpt), device_id=0, rank=0,
                      t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
    print(f"model ready in {time.perf_counter() - t_load:.1f}s", flush=True)

    def gen(c, prompt):
        return pipe.generate(prompt, size=(c["width"], c["height"]), frame_num=c["frames"], shift=c["shift"],
                             sample_solver=c["solver"], sampling_steps=c["steps"], guide_scale=c["cfg"],
                             seed=c["seed"], offload_model=c["offload"])

    # untimed warm-up with few steps: CUDA/kernel initialisation, not the 50-step cost
    gen(dict(BASE, steps=4), prompt_of[rows[0]["prompt_id"]])
    torch.cuda.synchronize()
    print("warmup done", flush=True)

    done = skipped = 0
    for suite in a.suites.split(","):
        for c in SUITES[suite]:
            d = os.path.join(out, c["name"])
            os.makedirs(d, exist_ok=True)
            for pid in subsets[c["subset"]]:
                mp4 = os.path.join(d, f"{pid}_{c['seed']}.mp4")
                rec = os.path.join(d, f"{pid}_{c['seed']}.json")
                if os.path.exists(mp4) and os.path.exists(rec):
                    skipped += 1
                    continue
                if time.monotonic() > deadline:
                    print(f"time budget reached; re-submit to continue. done={done} skipped={skipped}", flush=True)
                    return
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                video = gen(c, prompt_of[pid])
                torch.cuda.synchronize()
                secs = time.perf_counter() - t0
                peak_a, peak_r = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                tmp = mp4 + ".partial.mp4"
                cache_video(tensor=video[None], save_file=tmp, fps=FPS, nrow=1, normalize=True, value_range=(-1, 1))
                os.replace(tmp, mp4)
                record = {"schema": "genai-m1-v1", "method": "wan21_t2v_1.3b", "config_name": c["name"], "prompt_id": pid,
                          "seed": c["seed"], "config": c,
                          "video": {"frames": int(video.shape[1]), "height": int(video.shape[2]),
                                    "width": int(video.shape[3]), "fps": FPS, "sha256": sha256(mp4)},
                          "performance": {"success": True, "generation_seconds": secs,
                                          "scope": "WanT2V.generate: text encoding + denoising (with CFG) + VAE decode; "
                                                   "excludes model load and mp4 export",
                                          "peak_torch_allocated_bytes": peak_a, "peak_torch_reserved_bytes": peak_r,
                                          "memory_scope": "torch allocator peak inside generate (reset before each video)",
                                          "gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                          "cuda_build": torch.version.cuda},
                          "encoder": "wan.utils.utils.cache_video (imageio libx264 quality=8)"}
                json.dump(record, open(rec + ".tmp", "w"))
                os.replace(rec + ".tmp", rec)
                done += 1
                print(f"[{c['name']}] {pid} {secs:.1f}s peak={peak_a / 2**30:.1f}GiB frames={video.shape[1]}", flush=True)
    print(f"ALL DONE done={done} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
