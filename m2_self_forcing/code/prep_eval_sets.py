"""Build the folders VBench scores: 5 s videos (symlinks; `main` = the 130 main videos, `abl` = ablations and seeds) and, for 20 s rollouts, the first and last 4 s as clips."""
import argparse, glob, os
from multiprocessing import Pool

import imageio

SRC = os.path.expanduser("~/sfproj/outputs")
DST = os.path.expanduser("~/sfproj/eval_sets")
FPS, CLIP = 16, 64  # 4 s


def cut(task):
    path, cfg, base = task
    first = os.path.join(DST, "clips", f"{cfg}__first4s__{base}.mp4")
    last = os.path.join(DST, "clips", f"{cfg}__last4s__{base}.mp4")
    if os.path.exists(first) and os.path.exists(last):
        return 0
    frames = [f for f in imageio.get_reader(path)]
    for out, part in ((first, frames[:CLIP]), (last, frames[-CLIP:])):
        w = imageio.get_writer(out + ".tmp.mp4", fps=FPS, codec="libx264", quality=10, macro_block_size=1)
        for f in part:
            w.append_data(f)
        w.close()
        os.replace(out + ".tmp.mp4", out)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.join(DST, "main"), exist_ok=True)
    os.makedirs(os.path.join(DST, "abl"), exist_ok=True)
    os.makedirs(os.path.join(DST, "clips"), exist_ok=True)
    tasks, nshort = [], 0
    for cdir in sorted(glob.glob(os.path.join(SRC, "m2_*"))):
        cfg = os.path.basename(cdir)
        for f in sorted(glob.glob(os.path.join(cdir, "*.mp4"))):
            if f.endswith(".partial.mp4"):
                continue
            base = os.path.basename(f)[:-4]
            if "_long" in cfg:
                tasks.append((f, cfg, base))
            else:
                link = os.path.join(DST, "main" if cfg == "m2_main" else "abl", f"{cfg}__{base}.mp4")
                if not os.path.islink(link):
                    os.symlink(f, link)
                nshort += 1
    with Pool(a.procs) as p:
        done = sum(p.imap_unordered(cut, tasks))
    print(f"short links={nshort} long videos={len(tasks)} newly cut={done}", flush=True)


if __name__ == "__main__":
    main()
