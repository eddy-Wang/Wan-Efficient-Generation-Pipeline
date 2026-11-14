"""Light-weight evaluation for M2 videos: CLIP text alignment + per-2-second drift statistics.

CLIP follows the group definition: OpenAI ViT-B/32 (pinned revision), 16 uniformly sampled frames, original caption,
score = 100 * max(0, mean normalized cosine). Results are cached per video, so re-running only scores new videos.
"""
import argparse, glob, json, os, sys

import cv2
import numpy as np
import torch
from torchvision.io import read_video
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SEG_SECONDS = 2.0


def uniform_idx(n, k):
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


@torch.no_grad()
def embed(frames, model, proc, dev):
    px = proc(images=[f for f in frames], return_tensors="pt")["pixel_values"].to(dev)
    e = model.get_image_features(pixel_values=px).float()
    return torch.nn.functional.normalize(e, dim=-1)


def frame_stats(seg):
    """seg: T,H,W,C uint8. Returns brightness, saturation, sharpness (Laplacian var), motion (mean abs diff)."""
    idx = uniform_idx(len(seg), 8)
    br, sat, sharp = [], [], []
    for i in idx:
        f = seg[i]
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        br.append(float(g.mean()))
        sat.append(float(cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[..., 1].mean()))
        sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (208, 120)) for f in seg]).astype(np.float32)
    motion = float(np.abs(np.diff(small, axis=0)).mean()) if len(small) > 1 else 0.0
    return dict(brightness=np.mean(br), saturation=np.mean(sat), sharpness=np.mean(sharp), motion=motion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="outputs dir containing <config>/<pid>_<seed>.mp4")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="*")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    orig = {}
    for l in open(args.prompts, encoding="utf-8"):
        r = json.loads(l)
        orig[r["prompt_id"]] = r["original_prompt"]
    model = CLIPModel.from_pretrained(CLIP_ID, revision=CLIP_REV).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID, revision=CLIP_REV, use_fast=False)
    tok = CLIPTokenizer.from_pretrained(CLIP_ID, revision=CLIP_REV)
    os.makedirs(args.out, exist_ok=True)

    n_new = 0
    for cdir in sorted(glob.glob(os.path.join(args.videos, args.configs))):
        if not os.path.isdir(cdir):
            continue
        cname = os.path.basename(cdir)
        os.makedirs(os.path.join(args.out, cname), exist_ok=True)
        for mp4 in sorted(glob.glob(os.path.join(cdir, "*.mp4"))):
            if mp4.endswith(".partial.mp4"):
                continue
            stem = os.path.basename(mp4)[:-4]
            pid = stem.rsplit("_", 1)[0]
            res_path = os.path.join(args.out, cname, stem + ".json")
            if os.path.exists(res_path):
                continue
            video, _, info = read_video(mp4, pts_unit="sec", output_format="THWC")
            video = video.numpy()
            fps = float(info.get("video_fps", 16))
            caption = orig[pid]
            ntok = len(tok(caption, truncation=False)["input_ids"])
            with torch.no_grad():
                t_in = tok([caption], padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
                te = torch.nn.functional.normalize(model.get_text_features(**t_in).float(), dim=-1)
            seg_len = int(round(SEG_SECONDS * fps))
            nseg = max(1, len(video) // seg_len)
            segs = []
            first_emb = None
            for s in range(nseg):
                seg = video[s * seg_len:(s + 1) * seg_len] if nseg > 1 else video
                ie = embed([f for f in seg[uniform_idx(len(seg), 8)]], model, proc, dev)
                clip_text = float(100 * max(0.0, (ie @ te.T).mean().item()))
                m = torch.nn.functional.normalize(ie.mean(0, keepdim=True), dim=-1)
                if first_emb is None:
                    first_emb = m
                d = dict(segment=s, t_start=s * SEG_SECONDS, clip_text_x100=clip_text,
                         clip_to_first=float((m @ first_emb.T).item()))
                d.update({k: float(v) for k, v in frame_stats(seg).items()})
                segs.append(d)
            ie16 = embed([f for f in video[uniform_idx(len(video), 16)]], model, proc, dev)
            rec = dict(config=cname, prompt_id=pid, frames=int(len(video)), fps=fps,
                       height=int(video.shape[1]), width=int(video.shape[2]),
                       frame_mean_clip_cosine_x100=float(100 * max(0.0, (ie16 @ te.T).mean().item())),
                       caption_tokens=ntok, caption_truncated=bool(ntok > 77), segments=segs,
                       evaluator=dict(clip=f"{CLIP_ID}@{CLIP_REV}", device=dev, frames_for_clip=16))
            json.dump(rec, open(res_path + ".tmp", "w"))
            os.replace(res_path + ".tmp", res_path)
            n_new += 1
            if n_new % 20 == 0:
                print(f"scored {n_new} videos (last: {cname}/{stem})", flush=True)
    print(f"ANALYSIS DONE new={n_new}", flush=True)


if __name__ == "__main__":
    main()
