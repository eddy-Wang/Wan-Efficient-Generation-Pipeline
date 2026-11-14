"""Check that scoring a folder gives the same per-video values as scoring each video alone."""
import glob, json, os, sys
root = os.path.expanduser("~/sfproj/vbench_out")
batch = json.load(open(f"{root}/eq_batch_eval_results.json"))
worst = {}
for dim, (avg, rows) in batch.items():
    for r in rows:
        name = os.path.basename(r["video_path"])
        for f in glob.glob(f"{root}/eq_single_*_eval_results.json"):
            single = json.load(open(f))[dim][1]
            for q in single:
                if os.path.basename(q["video_path"]) == name:
                    a, b = r["video_results"], q["video_results"]
                    d = float(a != b) if isinstance(a, bool) else abs(float(a) - float(b))
                    worst[dim] = max(worst.get(dim, 0.0), d)
print("EQCHECK max abs difference batch vs single, per dimension:", json.dumps(worst))
