"""Small shared audio helpers (numpy only, so every environment can import them)."""
import numpy as np


def speech_bounds(y, sr, rel_db=-35.0, floor_margin_db=10.0, frame_ms=25, hop_ms=10, margin_s=0.05):
    """Start/end (seconds) of the part of a clip that contains speech.

    A frame counts as speech when its RMS is
      * within `rel_db` dB of the loudest frame (relative, so quiet and loud recordings
        are treated alike), and
      * at least `floor_margin_db` dB above the noise floor (10th percentile of frame
        levels), so clips with background noise are trimmed too.
    Leading and trailing silence is cut; `margin_s` is kept on each side.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(y) < frame:
        return 0.0, len(y) / sr
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = max(db.max() + rel_db, np.percentile(db, 10) + floor_margin_db)
    voiced = np.where(db >= thr)[0]
    if len(voiced) == 0:
        return 0.0, len(y) / sr
    start = max(voiced[0] * hop / sr - margin_s, 0.0)
    end = min((voiced[-1] * hop + frame) / sr + margin_s, len(y) / sr)
    return start, end
