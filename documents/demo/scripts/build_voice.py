"""Render the continuous voice-over (Kokoro neural TTS, voice af_heart) and its beat timings.

Reads assets/text/voiceover_v2.txt (one paragraph per beat), speaks every paragraph with the
same voice and speed, and joins them with short natural pauses (longer at a new act) into
one track. The video is then timed to this track (build_final_video.py).

Kokoro runs locally (model downloaded once from Hugging Face); it needs espeak-ng
(`brew install espeak-ng`) and Python 3.12:

    python3.12 -m venv .tts && .tts/bin/pip install "kokoro>=0.9.4" soundfile
    .tts/bin/python documents/demo/scripts/build_voice.py [--voice af_heart] [--speed 1.1]

Outputs assets/audio/voiceover_v2.wav and assets/audio/voiceover_v2_timings.json.
Defaults reproduce the final cut: af_heart blended 20% with af_nicole, speed 1.2,
0.15 s between sentences, 0.28 s between beats.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Point Kokoro's phonemizer at Homebrew's espeak-ng (the bundled espeakng-loader data is broken).
LIB, DATA = "/opt/homebrew/lib/libespeak-ng.dylib", "/opt/homebrew/share/espeak-ng-data"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"], os.environ["ESPEAK_DATA_PATH"] = LIB, DATA
import espeakng_loader  # noqa: E402

espeakng_loader.get_library_path = lambda: LIB
espeakng_loader.get_data_path = lambda: DATA
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402

EspeakWrapper.set_library(LIB)
EspeakWrapper.set_data_path(DATA)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from kokoro import KPipeline  # noqa: E402

DEMO = Path(__file__).resolve().parent.parent
SR = 24000
LEAD_IN, GAP, ACT_GAP = 0.5, 0.32, 0.65  # seconds of silence
ACT_START = {3, 6, 8, 9, 12, 15, 17, 22}  # beats that open a new act (longer breath before them)
HOLD = {4: 1.5, 7: 1.0, 8: 2.0, 9: 2.5, 16: 1.5, 20: 2.0, 21: 2.0, 22: 1.0}  # extra silence after dense slides


def trim(audio: np.ndarray, floor: float = 0.004) -> np.ndarray:
    """Drop the model's leading/trailing near-silence so our own pauses set the rhythm."""
    loud = np.flatnonzero(np.abs(audio) > floor)
    if not len(loud):
        return audio
    pad = int(0.03 * SR)
    return audio[max(0, loud[0] - pad): loud[-1] + pad]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--blend", default="af_nicole:0.2", help='softer mix, e.g. "af_nicole:0.2" (20%% of that voice)')
    ap.add_argument("--speed", type=float, default=1.2)
    ap.add_argument("--script", default="voiceover_v2.txt", help="file in assets/text/")
    ap.add_argument("--tag", default="_v2", help='output suffix, e.g. "_v2" -> voiceover_v2.wav')
    ap.add_argument("--sentence-gap", type=float, default=0.15,
                    help="render each sentence separately with this pause between them (v2); default: whole beat")
    ap.add_argument("--beat-gap", type=float, default=0.28, help="pause between beats; default v1 gaps + holds")
    args = ap.parse_args()
    lines = [l for l in (DEMO / "assets/text" / args.script).read_text().splitlines() if re.match(r"^\[\d+\]", l)]
    beats = [(int(re.match(r"^\[(\d+)\]", l).group(1)), re.sub(r"^\[\d+\]\s*", "", l)) for l in lines]
    pipe = KPipeline(lang_code="b" if args.voice.startswith("b") else "a", repo_id="hexgrad/Kokoro-82M")
    voice = args.voice
    if args.blend:  # weighted average of two voice embeddings: same voice, a touch softer
        other, w = args.blend.split(":")
        voice = (1 - float(w)) * pipe.load_voice(args.voice) + float(w) * pipe.load_voice(other)

    def speak(text: str) -> np.ndarray:
        return np.concatenate([a.numpy() for _, _, a in pipe(text, voice=voice, speed=args.speed)]).astype(np.float32)

    lead = 0.3 if args.beat_gap is not None else LEAD_IN
    parts, timings, t = [np.zeros(int(lead * SR), dtype=np.float32)], [], lead
    for i, (n, text) in enumerate(beats):
        if i:
            gap = args.beat_gap if args.beat_gap is not None else \
                (ACT_GAP if n in ACT_START else GAP) + HOLD.get(beats[i - 1][0], 0.0)
            parts.append(np.zeros(int(gap * SR), dtype=np.float32))
            t += gap
        if args.sentence_gap is None:
            audio = speak(text)
        else:
            sents = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text) if x]
            pause = np.zeros(int(args.sentence_gap * SR), dtype=np.float32)
            pieces = []
            for k, x in enumerate(sents):
                if k:
                    pieces.append(pause)
                pieces.append(trim(speak(x)))
            audio = np.concatenate(pieces)
        timings.append({"beat": n, "start": round(t, 3), "end": round(t + len(audio) / SR, 3), "text": text})
        parts.append(audio)
        t += len(audio) / SR
    out = DEMO / "assets/audio"
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / f"voiceover{args.tag}.wav", np.concatenate(parts), SR)
    (out / f"voiceover{args.tag}_timings.json").write_text(json.dumps(
        {"voice": args.voice, "blend": args.blend, "speed": args.speed, "length": round(t, 3), "beats": timings}, indent=1))
    print(f"{args.voice}{'+' + args.blend if args.blend else ''} @ {args.speed}: {t:.1f} s for {len(beats)} beats")


if __name__ == "__main__":
    main()
