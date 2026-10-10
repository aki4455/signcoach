"""Generate the English voice prompts as MP3 files with a local text-to-speech model (CLAUDE.md §5, §13).

Why pre-generated files: every phone then hears the same clear voice, and nothing is synthesised
(or sent anywhere) while the report plays. The browser's own speech is only a fallback.

- Texts: the "voice.*" keys in config/i18n/en.json (the website uses the same texts for the fallback);
  "voice.speed_limit" is expanded for every km/h value our class maps can output.
- Model: Piper TTS (GPL-3.0, optional `voice` dependency group, never installed on the server) with
  the en_US "ljspeech" voice (trained on the public-domain LJ Speech dataset), stored in
  $SIGNCOACH_DATA/models/piper/ with a pinned sha256.
- Output: assets/voice/en/<voice_key>.mp3 (mono, 64 kbit/s), plus silence.mp3 that the website plays
  once on the first tap to unlock audio on iPhones, and manifest.json (key -> text).

    uv run --group voice python scripts/generate_voice.py [--only KEY ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path

import yaml

from app.settings import get_settings

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "voice" / "en"
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/"
VOICE_FILES = {"en_US-ljspeech-high.onnx": "5d4f08ba6a2a48c44592eed3ce56bf85e9de3dd4e20df90541ae68a8310c029a",
               "en_US-ljspeech-high.onnx.json": None}
EVENT_VOICES = ["speeding", "stop_not_made", "turn_ban_violation"]
CONNECTORS = ["and"]  # joins two prompts due at once: "Main road… and pedestrian crossing."


def voice_texts() -> dict[str, str]:
    strings = json.loads((REPO / "config" / "i18n" / "en.json").read_text(encoding="utf-8"))
    catalogue = json.loads((REPO / "config" / "rules" / "signs.json").read_text(encoding="utf-8"))["signs"]
    values = set()
    for cmap in (REPO / "config" / "class_maps").glob("*.yaml"):
        values |= {v["value"] for v in yaml.safe_load(cmap.read_text())["classes"].values() if v.get("value")}
    texts = {}
    for key in {e["voice_key"] for e in catalogue if e.get("voice_key")} | set(EVENT_VOICES) | set(CONNECTORS):
        if "{value}" in key:
            for v in sorted(values):
                texts[key.format(value=v)] = strings[f"voice.{key.split('_{')[0]}"].format(value=v)
        else:
            texts[key] = strings[f"voice.{key}"]
    return dict(sorted(texts.items()))


def voice_model() -> Path:
    folder = get_settings().models_dir / "piper"
    folder.mkdir(parents=True, exist_ok=True)
    for name, sha in VOICE_FILES.items():
        path = folder / name
        if not path.exists():
            urllib.request.urlretrieve(VOICE_URL + name, path)
        if sha and hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"sha256 mismatch for {name}")
    return folder / "en_US-ljspeech-high.onnx"


def to_mp3(wav: Path, mp3: Path) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(wav), "-ac", "1", "-c:a", "libmp3lame",
                    "-b:a", "64k", str(mp3)], check=True)


def main() -> None:
    from piper import PiperVoice

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="voice keys to (re)generate; default: all")
    args = ap.parse_args()
    voice = PiperVoice.load(str(voice_model()))
    OUT.mkdir(parents=True, exist_ok=True)
    texts = voice_texts()
    with tempfile.TemporaryDirectory() as tmp:
        for key, text in texts.items():
            if args.only and key not in args.only:
                continue
            wav = Path(tmp) / f"{key}.wav"
            with wave.open(str(wav), "wb") as w:
                voice.synthesize_wav(text, w)
            to_mp3(wav, OUT / f"{key}.mp3")
        silence = Path(tmp) / "silence.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "anullsrc=r=22050:cl=mono", "-t", "0.3", str(silence)], check=True)
        to_mp3(silence, OUT / "silence.mp3")
    (OUT / "manifest.json").write_text(json.dumps(texts, indent=1) + "\n", encoding="utf-8")
    print(f"{len(texts)} prompts -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
