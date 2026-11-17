"""Step 2c: pilot F5-TTS-THAI on the pilot items (run on Windows with the GPU, env .venv-f5).

Model: VIZINTZOR/F5-TTS-THAI "v1" checkpoint (model_1000000.pt) + vocoder charactr/vocos-mel-24khz,
downloaded fresh from Hugging Face into checkpoints/hf (drive D). Inference code is the upstream
package f5-tts-th (MIT); MODEL_CFG is copied from f5_tts_th/tts.py for model "v1".

Two variants per item, upstream defaults otherwise (nfe_step 32, cfg 2.0, sway -1, target_rms 0.1):
  A  speed=1.0 (upstream default: length estimated from text)
  B  fix_duration = processed reference length + speech length of the real clip
     (silence at both ends of the real clip excluded; using the whole clip length made
     the model fill the extra time with words copied from the reference)

Output (local only, never commit): <out-dir>/<run-id>/
  <item_id>/real.wav, ref.wav, ref_processed.wav, clone_A.wav, clone_B.wav
  clones.csv, run.json, index.html
"""
import argparse
import html
import json
import os
import platform
import random
import shutil
import time
from importlib.metadata import version
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT / "checkpoints" / "hf"))
os.environ.setdefault("TORCH_HOME", str(PROJECT / "checkpoints" / "torch"))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from scipy.io import wavfile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import speech_bounds  # noqa: E402

MODEL_ID = "VIZINTZOR/F5-TTS-THAI"
CKPT_FILE = "model_1000000.pt"
VOCODER_ID = "charactr/vocos-mel-24khz"
SR = 24000
# copied from f5_tts_th/tts.py, model "v1"
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                 text_mask_padding=False, conv_layers=4, pe_attn_head=1)
UPSTREAM = dict(nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, target_rms=0.1,
                cross_fade_duration=0.15, set_max_chars=100, use_ipa=False)
VARIANTS = ("A", "B")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def mp3_to_pcm16(src: Path, dst: Path):
    """f5-tts-th reads references with pydub, which needs PCM WAV when ffmpeg is absent."""
    y, sr = sf.read(src, dtype="float32", always_2d=False)
    wavfile.write(dst, sr, np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16))
    return len(y) / sr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, type=Path)
    ap.add_argument("--cv-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("data/pilot/f5_tts_thai"), type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out = a.out_dir / a.run_id
    if out.exists():
        raise SystemExit(f"{out} already exists: choose a new --run-id")
    items = pd.read_csv(a.items, dtype={"item_id": str}, keep_default_na=False)
    if a.limit:
        items = items.head(a.limit)

    from f5_tts_th import utils_infer  # upstream package

    t0 = time.time()
    ckpt = hf_hub_download(MODEL_ID, CKPT_FILE)
    vocab = hf_hub_download(MODEL_ID, "vocab.txt")
    vocos_dir = snapshot_download(VOCODER_ID, allow_patterns=["config.yaml", "pytorch_model.bin"])
    download_sec = time.time() - t0
    t0 = time.time()
    vocoder = utils_infer.load_vocoder("vocos", is_local=True, local_path=vocos_dir, device=a.device)
    model = utils_infer.load_model(MODEL_CFG, ckpt, mel_spec_type="vocos", vocab_file=vocab, device=a.device)
    load_sec = time.time() - t0
    out.mkdir(parents=True)  # only after the model loaded
    cuda = a.device.startswith("cuda") and torch.cuda.is_available()
    quiet = lambda *x, **k: None  # noqa: E731

    rows = []
    for i, it in items.iterrows():
        d = out / it["item_id"]
        d.mkdir()
        wav, sr = sf.read(a.cv_dir / "clips" / it["real_path"], dtype="float32")
        sf.write(d / "real.wav", wav, sr)
        s0, s1 = speech_bounds(wav, sr)
        speech_sec = s1 - s0
        mp3_to_pcm16(a.cv_dir / "clips" / it["ref_path"], d / "ref.wav")
        proc_ref, proc_text = utils_infer.preprocess_ref_audio_text(
            str(d / "ref.wav"), it["ref_sentence"], show_info=quiet, device=a.device)
        shutil.copy(proc_ref, d / "ref_processed.wav")
        ref_proc_sec = sf.info(d / "ref_processed.wav").duration
        real_sec = float(it["real_sec"])
        for v in VARIANTS:
            seed = a.seed + i
            extra = {"speed": 1.0, "fix_duration": None} if v == "A" else \
                    {"speed": 1.0, "fix_duration": ref_proc_sec + speech_sec}
            row = {"item_id": it["item_id"], "variant": v, "seed": seed, "real_sec": real_sec,
                   "real_speech_sec": round(speech_sec, 3),
                   "ref_processed_sec": round(ref_proc_sec, 3), "fix_duration": extra["fix_duration"]}
            try:
                set_seed(seed)
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t = time.time()
                y, out_sr, _ = utils_infer.infer_process(
                    proc_ref, proc_text, it["real_sentence"], model, vocoder, mel_spec_type="vocos",
                    show_info=quiet, progress=None, device=a.device, **UPSTREAM, **extra)
                if cuda:
                    torch.cuda.synchronize()
                gen = time.time() - t
                assert out_sr == SR, out_sr
                y = np.asarray(y, dtype=np.float32)
                sf.write(d / f"clone_{v}.wav", y, SR)
                out_sec = len(y) / SR
                row.update(out_sec=round(out_sec, 3), gen_sec=round(gen, 2), rtf=round(gen / out_sec, 3),
                           peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
                           error="")
            except Exception as e:
                row.update(error=f"{type(e).__name__}: {e}"[:300])
                if cuda:
                    torch.cuda.empty_cache()
            rows.append(row)
            print(row)
        Path(proc_ref).unlink(missing_ok=True)

    clones = pd.DataFrame(rows)
    clones.to_csv(out / "clones.csv", index=False, encoding="utf-8")
    ok = clones[clones["error"] == ""]
    run = {
        "run_id": a.run_id, "model_id": MODEL_ID, "checkpoint": CKPT_FILE,
        "model_revision": Path(ckpt).parent.name, "vocoder_id": VOCODER_ID,
        "vocoder_revision": Path(vocos_dir).name, "model_cfg": MODEL_CFG, "upstream_params": UPSTREAM,
        "variants": {"A": "speed=1.0", "B": "fix_duration=ref_processed_sec+real_speech_sec"},
        "f5_tts_th_version": version("f5-tts-th"), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if cuda else None, "python": platform.python_version(),
        "items_file": str(a.items), "seed_base": a.seed,
        "download_sec": round(download_sec, 1), "load_sec": round(load_sec, 1),
        "clips_ok": int(len(ok)), "clips_failed": int(len(clones) - len(ok)),
        "median_gen_sec": float(ok["gen_sec"].median()) if len(ok) else None,
        "max_peak_vram_mb": float(ok["peak_vram_mb"].max()) if len(ok) and cuda else None,
        "median_abs_len_diff_vs_speech_sec": {v: float((g["out_sec"] - g["real_speech_sec"]).abs().median())
                                    for v, g in ok.groupby("variant")},
    }
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    cells = []
    for _, it in items.iterrows():
        iid = it["item_id"]
        aud = "".join(f'<td><audio controls preload="none" src="{iid}/{n}.wav"></audio></td>'
                      for n in ("real", "clone_A", "clone_B", "ref"))
        cells.append(f"<tr><td>{iid}</td><td>{html.escape(it['real_sentence'])}</td>{aud}</tr>")
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>F5-TTS-THAI pilot</title>"
        "<style>body{font-family:sans-serif}td{padding:4px;border-bottom:1px solid #ccc}</style>"
        f"<h3>F5-TTS-THAI pilot {html.escape(a.run_id)}</h3>"
        "<table><tr><th>item</th><th>ข้อความ</th><th>เสียงจริง</th><th>clone A (speed 1.0)</th>"
        "<th>clone B (ยาวเท่าช่วงพูดของเสียงจริง)</th><th>เสียงต้นแบบ</th></tr>" + "".join(cells) + "</table>",
        encoding="utf-8")
    print(json.dumps(run, ensure_ascii=False, indent=2))
    print(f"open {out / 'index.html'}")


if __name__ == "__main__":
    main()
