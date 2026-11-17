"""Step 1: choose the Common Voice cohort and split it by speaker.

Outputs (local only, never commit):
  <out-dir>/<run-id>/items.csv     one row per real clip, with its reference clip and VC source clip
  <out-dir>/<run-id>/summary.json  counts only (no IDs, no sentences)

Rules:
  * Listening-test pool, Train, Validation and Test share no speaker.
  * The paired sentence of a real clip never appears in another split.
  * Reference clip = another clip of the same speaker, different sentence, 2-8 s.
  * VC source clip = the same sentence read by a speaker who is not in any split.
  * Speakers and sentences from the old pilot are excluded.
  * Clips whose sentence contains the repetition mark "ๆ" or Latin letters are dropped
    (cloning systems skip or misread them), for both real and reference clips.
  * Clips whose sentence has fewer than --min-chars letters are dropped (very short texts
    make F5-family models pad the time with words copied from the reference).
"""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_ORDER = ["listen", "test", "validation", "train"]  # small sets pick sentences first


def text_key(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip().casefold()


def load_validated(cv_dir: Path) -> pd.DataFrame:
    v = pd.read_csv(cv_dir / "validated.tsv", sep="\t", quoting=csv.QUOTE_NONE,
                    dtype=str, keep_default_na=False)
    d = pd.read_csv(cv_dir / "clip_durations.tsv", sep="\t", dtype=str)
    d.columns = ["path", "ms"]
    v = v.merge(d, on="path", how="left")
    v["sec"] = pd.to_numeric(v["ms"], errors="coerce") / 1000.0
    v = v[(v["client_id"] != "") & v["sec"].notna()].drop_duplicates("path").copy()
    v["tkey"] = v["sentence"].map(text_key)
    return v


DEFAULT_DROP_PATTERN = r"[ๆA-Za-z]"


DEFAULT_MIN_CHARS = 8


def n_letters(s: str) -> int:
    """Letters in a sentence, not counting spaces, punctuation, tone marks or upper/lower vowels."""
    return len(re.sub(r"[\s\W_]", "", s))


def drop_short_text(v: pd.DataFrame, min_chars: int):
    """Remove clips whose sentence has fewer than min_chars letters; returns (kept, number dropped)."""
    if not min_chars:
        return v, 0
    short = v["sentence"].map(n_letters) < min_chars
    return v[~short].copy(), int(short.sum())


def drop_hard_text(v: pd.DataFrame, pattern: str):
    """Remove clips whose sentence matches pattern; returns (kept, number dropped)."""
    if not pattern:
        return v, 0
    bad = v["sentence"].str.contains(pattern, regex=True)
    return v[~bad].copy(), int(bad.sum())


def load_exclusions(paths):
    speakers, tkeys = set(), set()
    for p in paths:
        m = pd.read_csv(p, dtype=str, keep_default_na=False)
        if "client_id" in m:
            speakers |= set(m["client_id"]) - {""}
        col = "text_key" if "text_key" in m else ("sentence" if "sentence" in m else None)
        if col:
            tkeys |= {text_key(s) for s in m[col] if s}
    return speakers, tkeys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("data/cohort"), type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--exclude-manifest", action="append", default=[], type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-main", type=int, default=3000, help="real clips in Train+Validation+Test (1 per speaker)")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--n-listen-speakers", type=int, default=30)
    ap.add_argument("--listen-items-per-speaker", type=int, default=2)
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=10.0)
    ap.add_argument("--ref-min-sec", type=float, default=2.0)
    ap.add_argument("--ref-max-sec", type=float, default=8.0)
    ap.add_argument("--drop-pattern", default=DEFAULT_DROP_PATTERN,
                    help="regex; clips whose sentence matches are dropped ('' keeps all)")
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                    help="drop clips whose sentence has fewer letters than this (0 keeps all)")
    a = ap.parse_args()

    out = a.out_dir / a.run_id
    if out.exists():
        raise SystemExit(f"{out} already exists: choose a new --run-id (old runs are never overwritten)")

    rng = np.random.default_rng(a.seed)
    v = load_validated(a.cv_dir)
    n_all = len(v)
    ex_spk, ex_txt = load_exclusions(a.exclude_manifest)
    v = v[~v["client_id"].isin(ex_spk) & ~v["tkey"].isin(ex_txt)].copy()
    v, n_dropped_text = drop_hard_text(v, a.drop_pattern)
    v, n_dropped_short = drop_short_text(v, a.min_chars)

    paired_ok = v[(v["sec"] >= a.min_sec) & (v["sec"] <= a.max_sec)]
    ref_ok = v[(v["sec"] >= a.ref_min_sec) & (v["sec"] <= a.ref_max_sec)]
    paired_by = {k: g for k, g in paired_ok.groupby("client_id")}
    ref_by = {k: g for k, g in ref_ok.groupby("client_id")}

    def eligible(spk):
        p, r = paired_by.get(spk), ref_by.get(spk)
        if p is None or r is None:
            return False
        return len(set(p["tkey"]) | set(r["tkey"])) >= 2

    speakers = sorted(s for s in paired_by if eligible(s))
    speakers = list(rng.permutation(speakers))
    need = a.n_listen_speakers + a.n_main
    if need > len(speakers):
        raise SystemExit(f"need {need} speakers, only {len(speakers)} eligible")

    listen = speakers[: a.n_listen_speakers]
    main_spk = speakers[a.n_listen_speakers: need]
    unused = set(speakers[need:]) | (set(paired_by) - set(speakers))
    n_tr = round(a.ratios[0] * a.n_main)
    n_va = round(a.ratios[1] * a.n_main)
    groups = {"listen": listen, "train": main_spk[:n_tr],
              "validation": main_spk[n_tr:n_tr + n_va], "test": main_spk[n_tr + n_va:]}

    vc_pool = paired_ok[paired_ok["client_id"].isin(unused)]
    vc_by_tkey = {k: g["path"].tolist() for k, g in vc_pool.groupby("tkey")}

    owner = {}            # tkey -> split that owns it as a paired sentence
    used_in = {s: set() for s in groups}
    rows = []
    for split in SPLIT_ORDER:
        per_spk = a.listen_items_per_speaker if split == "listen" else 1
        for spk in groups[split]:
            cand = paired_by[spk]
            cand = cand[cand["tkey"].map(lambda t: owner.get(t, split) == split)]
            cand = cand.iloc[rng.permutation(len(cand))]
            cand = cand.assign(_vc=cand["tkey"].isin(vc_by_tkey).astype(int),
                               _dup=cand["tkey"].isin(used_in[split]).astype(int))
            cand = cand.sort_values(["_vc", "_dup"], ascending=[False, True], kind="stable")
            chosen, keys = [], set()
            for _, r in cand.iterrows():
                if r["tkey"] in keys:
                    continue
                refs = ref_by[spk]
                refs = refs[~refs["tkey"].isin(keys | {r["tkey"]}) & (refs["path"] != r["path"])]
                if refs.empty:
                    continue
                chosen.append((r, refs.iloc[int(rng.integers(len(refs)))]))
                keys.add(r["tkey"])
                if len(chosen) == per_spk:
                    break
            for r, ref in chosen:
                owner[r["tkey"]] = split
                used_in[split].add(r["tkey"])
                src = vc_by_tkey.get(r["tkey"], [])
                rows.append({
                    "split": split, "client_id": spk,
                    "real_path": r["path"], "real_sentence": r["sentence"], "real_sec": r["sec"],
                    "ref_path": ref["path"], "ref_sentence": ref["sentence"], "ref_sec": ref["sec"],
                    "vc_source_path": src[int(rng.integers(len(src)))] if src else "",
                    "gender": r["gender"], "age": r["age"], "tkey": r["tkey"],
                })

    items = pd.DataFrame(rows)
    items.insert(0, "item_id", [f"{s}_{i:05d}" for i, s in enumerate(items["split"])])

    # leakage checks
    spk_split = items.groupby("client_id")["split"].nunique()
    assert (spk_split == 1).all(), "speaker in more than one split"
    txt_split = items.groupby("tkey")["split"].nunique()
    assert (txt_split == 1).all(), "paired sentence in more than one split"
    assert (items["ref_path"] != items["real_path"]).all()
    assert not items["client_id"].isin(ex_spk).any() and not items["tkey"].isin(ex_txt).any()
    src_spk = v.set_index("path")["client_id"]
    vc_spk = items.loc[items["vc_source_path"] != "", "vc_source_path"].map(src_spk)
    assert not vc_spk.isin(set(items["client_id"])).any(), "VC source speaker is in a split"

    summary = {"run_id": a.run_id, "seed": a.seed, "cv_dir_name": a.cv_dir.name,
               "validated_rows": n_all, "excluded_speakers": len(ex_spk), "excluded_sentences": len(ex_txt),
               "dropped_clips_by_text_pattern": n_dropped_text, "drop_pattern": a.drop_pattern,
               "dropped_clips_short_text": n_dropped_short, "min_chars": a.min_chars,
               "eligible_speakers": len(speakers), "params": {k: (str(x) if isinstance(x, Path) else x)
                                                               for k, x in vars(a).items() if k not in ("cv_dir", "exclude_manifest")},
               "splits": {}}
    for split in SPLIT_ORDER:
        g = items[items["split"] == split]
        summary["splits"][split] = {
            "speakers": int(g["client_id"].nunique()), "real_clips": int(len(g)),
            "real_hours": round(float(g["real_sec"].sum()) / 3600, 3),
            "distinct_sentences": int(g["tkey"].nunique()),
            "vc_source_available": int((g["vc_source_path"] != "").sum()),
            "gender": {k or "unknown": int(n) for k, n in g["gender"].value_counts().items()},
            "real_sec_median": round(float(g["real_sec"].median()), 2),
        }
    out.mkdir(parents=True)
    items.drop(columns=["tkey"]).to_csv(out / "items.csv", index=False, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
