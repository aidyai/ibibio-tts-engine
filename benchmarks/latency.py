import argparse
import json
import statistics
import time
import urllib.request

WAV_HEADER = 44  # the stream sends a 44 byte wav header before any audio

TEXTS = {
    "short": "Ided owowaan ado akappa iwad",
    "medium": "Akpọ ifʌkkọ ado anie ewaana ye ọkpọ esịt ke nkọk itak isịn",
    "long": "Sake ado ukọd isọñ idʌñ Japanese enamma eto ibu-ibu adesi",
    "longer": "Akaanawaan ado amaase ntǝkayịn daña ebre mbre, atie ndobo-ndobo ke ñkpọitie inemesịt.",
    "mixed": "Kwa ndubi, eka ado ama edikood mme ñwed nnọ nditọ ammọ. But grandma is soo nice and I love her for that",
}


def post(url, path, text, voice):
    body = json.dumps({"text": text, "voice": voice}).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=body,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def stream_once(url, text, voice):
    # time to first audio = first byte after the wav header, not the header itself
    t0 = time.perf_counter()
    first_audio = None
    n = 0
    with post(url, "/synthesize/stream", text, voice) as r:
        while chunk := r.read1(65536):
            n += len(chunk)
            if first_audio is None and n > WAV_HEADER:
                first_audio = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return first_audio, total, max(n - WAV_HEADER, 0) / 48000  # 24khz, 16 bit, mono


def full_once(url, text, voice):
    t0 = time.perf_counter()
    with post(url, "/synthesize", text, voice) as r:
        n = len(r.read())
    return time.perf_counter() - t0, max(n - WAV_HEADER, 0) / 48000


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--voice", default="iniabasi")
    ap.add_argument("--cold", action="store_true", help="one request, no warm-up")
    ap.add_argument("--full", action="store_true", help="time /synthesize instead of the stream")
    a = ap.parse_args()

    if a.full and a.cold:
        t, secs = full_once(a.url, TEXTS["medium"], a.voice)
        print(f"cold: whole file after {t:.1f}s ({secs:.1f}s of audio)")
        return

    if a.cold:
        first, total, secs = stream_once(a.url, TEXTS["medium"], a.voice)
        if first is None:
            print("no audio came back, check the server logs")
            return
        print(f"cold: first audio after {first:.1f}s (total {total:.1f}s, {secs:.1f}s of audio)")
        return

    # first request is not counted, it can still be paying for a cold start
    if a.full:
        full_once(a.url, "Warm up.", a.voice)
        print(f"{'text':8} {'runs':>4} {'p50':>7} {'p95':>7} {'audio':>7}")
        for name, text in TEXTS.items():
            # the server caches by text, so add dots to make every run different
            rows = [full_once(a.url, text + "." * i, a.voice) for i in range(a.runs)]
            ts = [r[0] for r in rows]
            print(f"{name:8} {len(rows):>4} {statistics.median(ts):>6.2f}s {pct(ts, 95):>6.2f}s "
                  f"{statistics.median(r[1] for r in rows):>6.1f}s")
        return

    if stream_once(a.url, "Warm up.", a.voice)[0] is None:
        print("no audio came back, check the server logs. try --full to time /synthesize")
        return
    print(f"{'text':8} {'runs':>4} {'p50':>7} {'p95':>7} {'total':>7} {'audio':>7}")
    for name, text in TEXTS.items():
        rows = [stream_once(a.url, text, a.voice) for _ in range(a.runs)]
        firsts = [r[0] for r in rows if r[0] is not None]
        if not firsts:
            print(f"{name:8} no audio")
            continue
        print(f"{name:8} {len(rows):>4} {statistics.median(firsts):>6.2f}s {pct(firsts, 95):>6.2f}s "
              f"{statistics.median(r[1] for r in rows):>6.2f}s {statistics.median(r[2] for r in rows):>6.1f}s")
    print("p50 and p95 are time to first audio")


if __name__ == "__main__":
    main()
