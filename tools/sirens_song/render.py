"""Render the bilingual Sirens' Song dialogue video.

Picture: repo clip, re-timed shot by shot so cuts land in the speech pauses.
Audio:   isolated English dialogue only (no music, no original video audio).
Text:    one English line + one Chinese line per cue; speaker tags at turns.
"""
import argparse
import os
import re
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, SRC_FPS = 1080, 1920, 30, 60
DUR = 54.0
FONT_DIR = os.environ.get("FONT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"))
FONT_EN = os.path.join(FONT_DIR, "CormorantGaramond[wght].ttf")
FONT_ZH = os.path.join(FONT_DIR, "NotoSerifSC[wght].ttf")

# First frame of every shot in the source (60 fps), plus the end frame.
SHOT_STARTS = [0, 140, 324, 542, 756, 900, 1070, 1186, 1430, 1532, 1630, 1836, 1998, 2130, 2254, 2376]
# Output time each shot starts at; None = interpolate between neighbours.
# Fixed values put the cut inside a pause of the dialogue.
SHOT_TARGETS = [0.0, 3.18, 7.36, 12.68, 17.18, 21.02, None, 26.95, 32.79, None, 37.05, 41.73, None, 48.05, 51.23, DUR]

FADE_IN = (0.0, 0.6)
FADE_OUT = (52.6, 53.8)

TAG_EURY = ("EURYLOCHUS", "欧律罗科斯")
TAG_ODY = ("ODYSSEUS", "奥德修斯")

# (start, end, speaker tag or None, English, Chinese) — times on the output timeline.
# A cue whose speech ends before a nearby cut fades out on that cut instead of straddling it.
CUES = [
    (1.15, 3.18, TAG_EURY, "What was the Sirens’ song?", "塞壬之歌，唱的是什么？"),
    (4.05, 7.36, TAG_ODY, "All the things you’d want it to be.", "你想要它是什么，它就是什么。"),
    (7.75, 12.68, None, "Then all things you wish\nyou never wished for…", "然后，是你但愿从未奢望过的一切……"),
    (13.03, 17.18, None, "It was the delicious itch\nyou go to scratch…", "它是那阵撩人的痒，让你忍不住去挠……"),
    (17.87, 21.90, None, "But find it’s under the skin,\nand can’t be reached…", "却发现它在皮肉之下，怎么也够不着……"),
    (22.26, 26.49, None, "So the delicious itch\nbecomes unbearable…", "于是，那撩人的痒，变得难以忍受……"),
    (27.14, 33.62, None, "It told you what you most want\nis what you most can’t have,", "它告诉你：你最想要的，就是你最得不到的；"),
    (34.18, 41.08, None, "and what you most can’t have\nis what you already had and lost…", "而最得不到的，是你曾经拥有、却已失去的……"),
    (43.44, 47.88, None, "It was the song of all the promises\nI’ve failed to keep…", "它唱的，是我所有未能兑现的诺言……"),
    (48.18, FADE_OUT[1], TAG_ODY, "And it told me\nI don’t really want to go home.", "它还告诉我：其实，我并不想回家。"),
]

SUB_FADE_IN, SUB_FADE_OUT, SUB_RISE = 0.22, 0.30, 10
BLOCK_BOTTOM = 1545
EN_SIZE, ZH_SIZE, TAG_SIZE = 62, 44, 30
EN_MAX_W, ZH_MAX_W = 940, 950
WHITE = (247, 243, 234)
ZH_COLOR = (236, 229, 214)
GOLD = (216, 186, 128)


def shot_targets():
    t = list(SHOT_TARGETS)
    for i, v in enumerate(t):
        if v is None:
            lo = max(j for j in range(i) if t[j] is not None)
            hi = min(j for j in range(i + 1, len(t)) if t[j] is not None)
            rate = (t[hi] - t[lo]) / (SHOT_STARTS[hi] - SHOT_STARTS[lo])
            t[i] = t[lo] + (SHOT_STARTS[i] - SHOT_STARTS[lo]) * rate
    return t


def src_position(t, targets):
    """Fractional source frame index for output time t, and the shot's frame range."""
    k = max(i for i in range(len(targets) - 1) if targets[i] <= t) if t >= 0 else 0
    k = min(k, len(targets) - 2)
    s0, s1 = SHOT_STARTS[k], SHOT_STARTS[k + 1]
    frac = (t - targets[k]) / (targets[k + 1] - targets[k])
    p = s0 + frac * (s1 - s0) - 0.5
    return min(max(p, s0), s1 - 1), s0, s1


def font(path, size, weight):
    f = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_axes([weight])
    return f


def text_w(draw, s, f, tracking=0):
    return draw.textlength(s, font=f) + tracking * max(len(s) - 1, 0)


def wrap_en(draw, s, f, max_w):
    if "\n" in s:  # hand-set phrase breaks
        lines = s.split("\n")
        assert all(text_w(draw, l, f) <= max_w for l in lines), s
        return lines
    if text_w(draw, s, f) <= max_w:
        return [s]
    words = s.split()
    best = None
    for i in range(1, len(words)):  # balanced two-line split
        a, b = " ".join(words[:i]), " ".join(words[i:])
        wa, wb = text_w(draw, a, f), text_w(draw, b, f)
        if max(wa, wb) <= max_w:
            score = abs(wa - wb) - (40 if a.endswith(",") else 0)
            if best is None or score < best[0]:
                best = (score, [a, b])
    return best[1]


def wrap_zh(draw, s, f, max_w):
    if text_w(draw, s, f) <= max_w:
        return [s]
    cuts = [i + 1 for i, c in enumerate(s) if c in "，、：；" and i + 1 < len(s)]
    i = min(cuts, key=lambda c: abs(text_w(draw, s[:c], f) - text_w(draw, s[c:], f)))
    return [s[:i], s[i:]]


def draw_tracked(draw, xy, s, f, fill, tracking):
    x, y = xy
    for ch in s:
        draw.text((x, y), ch, font=f, fill=fill)
        x += draw.textlength(ch, font=f) + tracking


def render_block(tag, en, zh):
    """Return (premultiplied rgb float32, alpha float32, top y) for one cue."""
    f_en = font(FONT_EN, EN_SIZE, 600)
    f_zh = font(FONT_ZH, ZH_SIZE, 560)
    f_tag = font(FONT_EN, TAG_SIZE, 700)
    f_tag_zh = font(FONT_ZH, TAG_SIZE - 4, 600)
    scratch = ImageDraw.Draw(Image.new("L", (10, 10)))
    en_lines = wrap_en(scratch, en, f_en, EN_MAX_W)
    zh_lines = wrap_zh(scratch, zh, f_zh, ZH_MAX_W)
    assert len(zh_lines) == 1, zh

    en_lh, zh_lh = int(EN_SIZE * 1.16), int(ZH_SIZE * 1.5)
    tag_h, tag_gap, mid_gap = 36, 22, 18
    h = (tag_h + tag_gap if tag else 0) + en_lh * len(en_lines) + mid_gap + zh_lh * len(zh_lines)
    pad = 40
    bh = h + 2 * pad
    color = Image.new("RGB", (W, bh), (0, 0, 0))
    mask = Image.new("L", (W, bh), 0)
    cd, md = ImageDraw.Draw(color), ImageDraw.Draw(mask)

    def put(x, y, s, f, fill, tracking=0):
        if tracking:
            draw_tracked(cd, (x, y), s, f, fill, tracking)
            draw_tracked(md, (x, y), s, f, 255, tracking)
        else:
            cd.text((x, y), s, font=f, fill=fill)
            md.text((x, y), s, font=f, fill=255)

    y = pad
    if tag:
        name, zh_name = tag
        tr = 7
        wn = text_w(scratch, name, f_tag, tr)
        wz = text_w(scratch, zh_name, f_tag_zh, 4)
        dot_gap = 46
        total = wn + dot_gap + wz
        x0 = (W - total) / 2
        put(x0, y + 2, name, f_tag, GOLD, tr)
        put(x0 + wn + dot_gap, y + 1, zh_name, f_tag_zh, GOLD, 4)
        cx = x0 + wn + dot_gap / 2 + 3
        for d in (cd, md):
            d.ellipse((cx - 3, y + 18 - 3, cx + 3, y + 18 + 3), fill=GOLD if d is cd else 255)
        y += tag_h + tag_gap
    for line in en_lines:
        put((W - text_w(scratch, line, f_en)) / 2, y, line, f_en, WHITE)
        y += en_lh
    y += mid_gap
    for line in zh_lines:
        x = (W - text_w(scratch, line, f_zh)) / 2
        for part in re.split("(……)", line):
            dy = -round(ZH_SIZE * 0.27) if part == "……" else 0
            put(x, y + dy, part, f_zh, ZH_COLOR)
            x += text_w(scratch, part, f_zh)
        y += zh_lh

    m = np.asarray(mask, np.float32) / 255
    rgb = np.asarray(color, np.float32)
    # Two soft shadows: a wide halo for bright backgrounds, a tight one for edge definition.
    wide = np.asarray(mask.filter(ImageFilter.GaussianBlur(14)), np.float32) / 255
    tight = np.asarray(mask.filter(ImageFilter.GaussianBlur(3)).transform(
        mask.size, Image.AFFINE, (1, 0, 0, 0, 1, -2)), np.float32) / 255
    shadow = 1 - (1 - np.minimum(wide * 0.9, 0.55)) * (1 - tight * 0.6)
    alpha = m + shadow * (1 - m)
    premult = rgb * m[..., None]  # shadow contributes black
    top = BLOCK_BOTTOM + pad - bh
    return premult, alpha, top


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def scrim():
    y = np.arange(H, dtype=np.float32)
    a = np.clip((y - 980) / (H - 980), 0, 1)
    a = a * a * (3 - 2 * a) * 0.62
    return (1 - a)[:, None, None]


def frame_reader(path):
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    size = W * H * 3
    idx = 0
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield idx, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stills", help="comma separated output times: write PNGs instead of video")
    args = ap.parse_args()

    targets = shot_targets()
    for i in range(len(targets) - 1):
        span = (SHOT_STARTS[i + 1] - SHOT_STARTS[i]) / SRC_FPS
        print(f"shot {i:2d}  {targets[i]:6.2f}-{targets[i + 1]:6.2f}  speed {span / (targets[i + 1] - targets[i]):.3f}x")

    blocks = [render_block(tag, en, zh) for _, _, tag, en, zh in CUES]
    sc = scrim()

    times = ([float(s) for s in args.stills.split(",")] if args.stills
             else [(j + 0.5) / FPS for j in range(int(round(DUR * FPS)))])

    enc = None
    if not args.stills:
        enc = subprocess.Popen([
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-i", args.audio,
            "-map", "0:v", "-map", "1:a",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-level", "4.2",
            "-tune", "film", "-g", str(FPS * 2), "-bf", "2",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", "-t", f"{DUR}", args.out,
        ], stdin=subprocess.PIPE)

    reader = frame_reader(args.src)
    cache = {}
    last_read = -1

    def get(i):
        nonlocal last_read
        while last_read < i:
            last_read, fr = next(reader)
            cache[last_read] = fr
        for k in [k for k in cache if k < i - 2]:
            del cache[k]
        return cache[i]

    for n, t in enumerate(times):
        p, s0, s1 = src_position(t, targets)
        i0 = int(np.floor(p))
        f = p - i0
        a = get(i0).astype(np.float32)
        if f > 1e-3 and i0 + 1 < s1:
            a = a * (1 - f) + get(i0 + 1).astype(np.float32) * f
        img = a * sc

        for (start, end, *_), (prem, alpha, top) in zip(CUES, blocks):
            if not (start <= t < end):
                continue
            fade_out = SUB_FADE_OUT if end < FADE_OUT[1] else 1e-6
            k = smooth((t - start) / SUB_FADE_IN) * smooth((end - t) / fade_out)
            if k <= 0:
                continue
            dy = int(round((1 - smooth((t - start) / (SUB_FADE_IN * 1.6))) * SUB_RISE))
            y0 = top + dy
            ys, ye = max(y0, 0), min(y0 + alpha.shape[0], H)
            al = alpha[ys - y0:ye - y0, :, None] * k
            img[ys:ye] = img[ys:ye] * (1 - al) + prem[ys - y0:ye - y0] * k

        g = smooth((t - FADE_IN[0]) / (FADE_IN[1] - FADE_IN[0])) * (1 - smooth((t - FADE_OUT[0]) / (FADE_OUT[1] - FADE_OUT[0])))
        if g < 1:
            img = img * g
        out = np.clip(img + 0.5, 0, 255).astype(np.uint8)

        if args.stills:
            Image.fromarray(out).save(f"{args.out}_{t:05.2f}.png")
        else:
            enc.stdin.write(out.tobytes())
            if n % 150 == 0:
                print(f"frame {n}/{len(times)}", flush=True)

    if enc:
        enc.stdin.close()
        enc.wait()


if __name__ == "__main__":
    main()
