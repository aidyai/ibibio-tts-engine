import argparse
import csv
import os
import time

from latency import TEXTS, stream_once, full_once, post

VOICES = ["ama", "iniabasi"]


def save_wav(url, text, voice, path):
    with post(url, "/synthesize", text, voice) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="../docs/samples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # warm up so the first row is not a cold start
    full_once(a.url, "Warm up.", VOICES[0])

    rows = []
    for voice in VOICES:
        for name, text in TEXTS.items():
            first, total, secs = stream_once(a.url, text, voice)
            path = os.path.join(a.out, f"{voice}_{name}.wav")
            save_wav(a.url, text, voice, path)
            rows.append([voice, name, text, f"{first:.2f}" if first else "none", f"{secs:.1f}"])
            print(f"{voice:9} {name:7} first audio {rows[-1][3]:>5}s  audio {secs:.1f}s  -> {path}")

    with open(os.path.join(a.out, "samples.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["voice", "key", "text", "first_audio_s", "audio_s"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
