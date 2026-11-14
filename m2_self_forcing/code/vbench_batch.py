"""Score every mp4 in one folder with the pinned VBench source (custom_input mode), loading each model once.

Same code revision and dependency constraints as the group's per-video evaluator; the five dimensions used are
computed per video, so scoring a folder is equivalent to scoring the videos one by one.
"""
import argparse, json, os, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pinned VBench checkout")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dims", nargs="+", default=["subject_consistency", "motion_smoothness", "dynamic_degree",
                                                  "imaging_quality", "aesthetic_quality"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import subprocess
    rev = subprocess.check_output(["git", "-C", a.src, "rev-parse", "HEAD"], text=True).strip()
    assert rev == "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490", f"unexpected VBench revision {rev}"
    sys.path.insert(0, a.src)
    import torch
    from vbench import VBench
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / f"{a.name}_eval_results.json"
    n = len([f for f in os.listdir(a.videos) if f.endswith(".mp4")])
    if res_path.exists():
        print(f"SKIP: {res_path} exists", flush=True)
        return
    t = time.perf_counter()
    ev = VBench(torch.device(a.device), str(Path(a.src) / "vbench/VBench_full_info.json"), str(out))
    ev.evaluate(videos_path=a.videos, name=a.name, dimension_list=a.dims, mode="custom_input")
    print(f"VBENCH DONE videos={n} dims={a.dims} seconds={time.perf_counter() - t:.1f}", flush=True)


if __name__ == "__main__":
    main()
