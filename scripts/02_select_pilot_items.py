"""Step 2a: choose pilot items for trying a cloning system.

Pilot speakers and pilot sentences are kept out of every split of the cohort
(listen / train / validation / test) and out of the old project pilot, so the
pilot never touches data used later.

Output (local only): <out-dir>/<run-id>/pilot_items.csv and summary.json
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("step1", HERE / "01_select_cohort_split.py")
step1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-dir", required=True, type=Path)
    ap.add_argument("--cohort", required=True, type=Path, help="data/cohort/<run-id>")
    ap.add_argument("--exclude-manifest", action="append", default=[], type=Path)
    ap.add_argument("--out-dir", default=Path("data/pilot_items"), type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-sec", type=float, default=2.0)
    ap.add_argument("--max-sec", type=float, default=8.0)
    ap.add_argument("--drop-pattern", default=step1.DEFAULT_DROP_PATTERN)
    ap.add_argument("--min-chars", type=int, default=step1.DEFAULT_MIN_CHARS)
    a = ap.parse_args()

    out = a.out_dir / a.run_id
    if out.exists():
        raise SystemExit(f"{out} already exists: choose a new --run-id")

    v = step1.load_validated(a.cv_dir)
    cohort = pd.read_csv(a.cohort / "items.csv", dtype=str, keep_default_na=False)
    ex_spk, ex_txt = step1.load_exclusions(a.exclude_manifest)
    ex_spk |= set(cohort["client_id"])
    ex_txt |= {step1.text_key(s) for s in cohort["real_sentence"]}
    used_paths = set(cohort["vc_source_path"]) - {""}

    v = v[~v["client_id"].isin(ex_spk) & ~v["tkey"].isin(ex_txt) & ~v["path"].isin(used_paths)]
    v = v[(v["sec"] >= a.min_sec) & (v["sec"] <= a.max_sec)]
    v, _ = step1.drop_hard_text(v, a.drop_pattern)
    v, _ = step1.drop_short_text(v, a.min_chars)

    rng = np.random.default_rng(a.seed)
    rows = []
    for spk in rng.permutation(sorted(v["client_id"].unique())):
        g = v[v["client_id"] == spk]
        if g["tkey"].nunique() < 2:
            continue
        g = g.iloc[rng.permutation(len(g))]
        real = g.iloc[0]
        ref = g[g["tkey"] != real["tkey"]].iloc[0]
        rows.append({"item_id": f"pilot_{len(rows):03d}", "client_id": spk,
                     "real_path": real["path"], "real_sentence": real["sentence"], "real_sec": real["sec"],
                     "ref_path": ref["path"], "ref_sentence": ref["sentence"], "ref_sec": ref["sec"],
                     "gender": real["gender"]})
        if len(rows) == a.n:
            break

    items = pd.DataFrame(rows)
    out.mkdir(parents=True)
    items.to_csv(out / "pilot_items.csv", index=False, encoding="utf-8")
    summary = {"run_id": a.run_id, "seed": a.seed, "cohort": a.cohort.name, "items": len(items),
               "speakers": int(items["client_id"].nunique()),
               "real_sec_median": round(float(items["real_sec"].median()), 2),
               "ref_sec_median": round(float(items["ref_sec"].median()), 2)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
