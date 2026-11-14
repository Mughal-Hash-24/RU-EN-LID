"""Draw the proposed-architecture diagrams as SVG (standard library only).

    python scripts/make_architecture_diagrams.py        # writes docs/architecture/*.svg

PNG copies for slides (docs/architecture/png/) are exported from the SVGs with a browser
at 2x; any SVG-to-PNG converter works.

Diagrams are code so they stay true to the pipeline: the running-example trace in
overview.svg is computed by running the model-free pipeline (context penalty 1.0),
and the lexicon counts are read from data/resources. Re-run after changing either.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier.features import word_features  # noqa: E402
from src.pipeline import DEFAULT_RESOURCES, LIDPipeline  # noqa: E402
from src.verification.rules import DEFAULT_TAU  # noqa: E402

OUT = ROOT / "docs" / "architecture"
SANS = "Inter, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"
W = 1600

# ---- palette -------------------------------------------------------------------------------
PAPER, CARD, HAIR, INK, MUTED, FAINT = "#FBFAF6", "#FFFFFF", "#E5E2DA", "#1E2230", "#5E6675", "#8A919E"
STAGE = {
    1: ("#4338A0", "#EFEDFB"),  # indigo
    2: ("#0F7B70", "#E5F4F1"),  # teal
    3: ("#9C2F5E", "#F9EAF0"),  # plum
}
GATE, GATE_L = "#B45309", "#FDF2E4"
TAG = {  # fill, stroke, text
    "EN": ("#E4EDFC", "#2F62C0", "#1D4594"),
    "UR": ("#E2F4E9", "#2C8A55", "#1C6440"),
    "AMB": ("#FDF1DA", "#D4870E", "#8A5300"),
    "VAR": ("#EFE9FC", "#7350C6", "#4F3196"),
    "BOTH": ("url(#both)", "#9A5A86", "#5B2E52"),
    "UNK": ("#F3F4F6", "#9AA0A6", "#5F6368"),
    "OTH": ("#F3F4F6", "#9AA0A6", "#5F6368"),
}
TAG_MEANING = {
    "EN": "English lexicon only",
    "UR": "Roman Urdu lexicon only",
    "AMB": "in both lexicons",
    "VAR": "attested spelling variant",
    "BOTH": "variant that is also English",
    "UNK": "in no lexicon",
}


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, height: int, title: str, desc: str) -> None:
        self.h = height
        self.parts: list[str] = []
        self.title, self.desc = title, desc

    def add(self, s: str) -> None:
        self.parts.append(s)

    # ---- primitives ------------------------------------------------------------------------
    def rect(self, x, y, w, h, fill=CARD, stroke=HAIR, sw=1.0, r=12, dash=None, shadow=False, opacity=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += ' filter="url(#shadow)"' if shadow else ""
        extra += f' opacity="{opacity}"' if opacity is not None else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def text(self, x, y, s, size=14, weight=400, fill=INK, anchor="start", mono=False, italic=False, ls=None):
        family = MONO if mono else SANS
        extra = ' font-style="italic"' if italic else ""
        extra += f' letter-spacing="{ls}"' if ls is not None else ""
        self.add(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{extra}>{esc(s)}</text>'
        )

    def rich(self, x, y, runs, size=14, anchor="start"):
        """runs: list of (text, dict(weight, fill, mono, italic))."""
        spans = []
        for s, style in runs:
            family = MONO if style.get("mono") else SANS
            attrs = f'font-family="{family}" font-weight="{style.get("weight", 400)}" fill="{style.get("fill", INK)}"'
            if style.get("italic"):
                attrs += ' font-style="italic"'
            spans.append(f"<tspan {attrs}>{esc(s)}</tspan>")
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{"".join(spans)}</text>')

    def line(self, points, color=INK, sw=1.6, arrow=True, dash=None, marker=None):
        d = "M " + " L ".join(f"{px} {py}" for px, py in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if arrow:
            extra += f' marker-end="url(#{marker or self.marker(color)})"'
        self.add(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"{extra}/>'
        )

    _markers: dict[str, str] = {}

    def marker(self, color: str) -> str:
        key = "ah" + color.strip("#")
        Svg._markers[key] = color
        return key

    def chip(self, cx, cy, tag, label=None, w=None, h=26, size=12.5, mono=True, dashed=False, faded=False, ring=None):
        fill, stroke, ink = TAG[tag]
        label = tag if label is None else label
        w = w or max(44, int(len(label) * size * 0.62) + 18)
        if ring:
            self.rect(cx - w / 2 - 4, cy - h / 2 - 4, w + 8, h + 8, fill="none", stroke=ring, sw=2, r=h / 2 + 4)
        self.rect(
            cx - w / 2, cy - h / 2, w, h, fill=fill, stroke=stroke, sw=1.3, r=h / 2,
            dash="4 3" if dashed or tag == "UNK" else None, opacity=0.62 if faded else None,
        )
        self.text(cx, cy + size * 0.36, label, size=size, weight=600, fill=ink, anchor="middle", mono=mono)
        return w

    def stage_card(self, x, y, w, h, n, title, eyebrow=None):
        accent, light = STAGE[n]
        self.rect(x, y, w, h, shadow=True, r=14)
        self.add(
            f'<path d="M {x} {y + 14} a 14 14 0 0 1 14 -14 h {w - 28} a 14 14 0 0 1 14 14 v 46 h {-w} z" '
            f'fill="{light}"/>'
        )
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" stroke="{HAIR}"/>')
        self.text(x + 20, y + 25, eyebrow or f"STAGE {n}", size=11.5, weight=700, fill=accent, ls=1.6)
        self.text(x + 20, y + 47, title, size=17, weight=700, fill=INK)

    def plain_card(self, x, y, w, h, eyebrow, title, accent=MUTED, dashed=False):
        self.rect(x, y, w, h, shadow=not dashed, r=14, dash="5 4" if dashed else None,
                  stroke=FAINT if dashed else HAIR, fill=CARD if not dashed else "#FFFFFFCC")
        self.text(x + 20, y + 25, eyebrow, size=11.5, weight=700, fill=accent, ls=1.6)
        if title:
            self.text(x + 20, y + 47, title, size=17, weight=700, fill=INK)

    def header(self, title, subtitle, page):
        self.text(60, 66, title, size=30, weight=700)
        self.text(60, 98, subtitle, size=16, fill=MUTED)
        self.text(W - 60, 64, "RU-EN-LID", size=13, weight=700, fill=INK, anchor="end", ls=1.2)
        self.text(W - 60, 84, f"proposed architecture · {page}", size=12, fill=FAINT, anchor="end")
        self.add(f'<line x1="60" y1="122" x2="{W - 60}" y2="122" stroke="{HAIR}" stroke-width="1"/>')

    def render(self) -> str:
        markers = "".join(
            f'<marker id="{key}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" '
            f'orient="auto-start-reverse" markerUnits="userSpaceOnUse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
            for key, color in Svg._markers.items()
        )
        defs = (
            "<defs>"
            '<filter id="shadow" x="-5%" y="-5%" width="110%" height="120%">'
            f'<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="{INK}" flood-opacity="0.07"/></filter>'
            '<linearGradient id="both" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0.5" stop-color="{TAG["AMB"][0]}"/><stop offset="0.5" stop-color="{TAG["VAR"][0]}"/>'
            "</linearGradient>"
            '<pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="1" fill="#E7E4DC"/></pattern>'
            f"{markers}</defs>"
        )
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.h}" '
            f'viewBox="0 0 {W} {self.h}" role="img" aria-labelledby="t d">\n'
            f'<title id="t">{esc(self.title)}</title><desc id="d">{esc(self.desc)}</desc>\n{defs}\n'
            f'<rect width="{W}" height="{self.h}" fill="{PAPER}"/>'
            f'<rect width="{W}" height="{self.h}" fill="url(#dots)"/>\n{body}\n</svg>\n'
        )


# ---- data drawn from the code ----------------------------------------------------------------
EXAMPLE_ID = "s1"
WINDOW = slice(6, 17)  # "box band nhi tha main bht upset hoon is bar bhi"


def running_example():
    rows = [r for r in csv.DictReader(open(ROOT / "data/demo/running_examples.csv", encoding="utf-8"))
            if r["sentence_id"] == EXAMPLE_ID]
    words = [r["token"] for r in rows]
    gold = [r["label"] for r in rows]
    pipeline = LIDPipeline.from_resources(context_penalty=1.0)
    records = pipeline.run_records(" ".join(words))
    tags, readings = word_features(pipeline.expander, words)
    return words[WINDOW], tags[WINDOW], readings[WINDOW], records[WINDOW], gold[WINDOW]


def lexicon_counts():
    expander = LIDPipeline.from_resources().expander
    pairs = [line for line in open(Path(DEFAULT_RESOURCES) / "variant_pairs.tsv", encoding="utf-8")
             if line.strip() and not line.startswith("#")]
    return len(expander.lexicons.english), len(expander.lexicons.roman_urdu), len(pairs)


# ---- overview ----------------------------------------------------------------------------------
def overview() -> str:
    n_en, n_ur, n_pairs = lexicon_counts()
    words, tags, readings, records, gold = running_example()
    s = Svg(1060, "RU-EN-LID proposed architecture: overview",
            "Three stages: candidate expansion tags each word, a candidate-conditioned XLM-R labels it with a "
            "confidence, a confidence gate keeps confident words and sends the rest to rule- and context-based "
            "verification. A running example is traced word by word.")
    s.header("Proposed architecture", "Roman Urdu–English word-level language identification. Each stage adds "
             "evidence; no stage rewrites a word.", "overview")

    top, h = 176, 268
    # input
    s.plain_card(60, top, 176, h, "INPUT", "Sentence")
    for i, chunk in enumerate(["box band nhi", "tha main bht", "upset hoon is", "bar bhi …"]):
        s.text(80, top + 92 + i * 30, chunk, size=15, mono=True, weight=500)
    s.text(80, top + 222, "romanized Urdu +", size=12.5, fill=MUTED)
    s.text(80, top + 240, "English, no standard", size=12.5, fill=MUTED)
    s.text(80, top + 258 - 2, "spelling", size=12.5, fill=MUTED)

    # stage 1
    x1 = 286
    s.stage_card(x1, top, 300, h, 1, "Candidate expansion")
    s.text(x1 + 20, top + 92, "Looks every word up:", size=14, weight=600)
    s.text(x1 + 20, top + 116, f"• English lexicon ({n_en} seed words)", size=13.5, fill=MUTED)
    s.text(x1 + 20, top + 138, f"• Roman Urdu lexicon ({n_ur})", size=13.5, fill=MUTED)
    s.text(x1 + 20, top + 160, f"• {n_pairs} attested spelling variants", size=13.5, fill=MUTED)
    s.text(x1 + 20, top + 190, "Adds one tag + an Urdu reading.", size=13.5, weight=600)
    for i, tag in enumerate(("EN", "UR", "AMB", "VAR", "BOTH", "UNK")):
        s.chip(x1 + 40 + i * 44, top + 228, tag, size=10.5, h=24, w=40)
    # stage 2
    x2 = 636
    s.stage_card(x2, top, 300, h, 2, "Candidate-conditioned XLM-R")
    s.text(x2 + 20, top + 92, "Input embedding of each word piece =", size=14, weight=600)
    s.rich(x2 + 20, top + 118, [("word piece ", {"mono": True, "weight": 600}),
                                 ("+ ", {"fill": MUTED}), ("tag", {"mono": True, "weight": 600, "fill": TAG["VAR"][2]}),
                                 (" + g·", {"fill": MUTED}),
                                 ("reading", {"mono": True, "weight": 600, "fill": TAG["VAR"][2]})], size=13.5)
    s.text(x2 + 20, top + 148, "12 layers read the whole sentence", size=13.5, fill=MUTED)
    s.text(x2 + 20, top + 170, "and weigh lexicon vs context.", size=13.5, fill=MUTED)
    s.text(x2 + 20, top + 200, "Output for every word:", size=14, weight=600)
    for i, lab in enumerate(("UR", "EN", "OTH")):
        s.chip(x2 + 44 + i * 54, top + 232, lab, size=11, h=24, w=46)
    s.text(x2 + 186, top + 236, "+ confidence", size=13, fill=MUTED)

    # gate
    gx, gy, gr = 1006, top + h / 2, 50
    s.add(f'<path d="M {gx} {gy - gr} L {gx + gr} {gy} L {gx} {gy + gr} L {gx - gr} {gy} z" '
          f'fill="{GATE_L}" stroke="{GATE}" stroke-width="1.6" filter="url(#shadow)"/>')
    s.text(gx, gy - 4, "conf ≥ τ ?", size=13.5, weight=700, fill=GATE, anchor="middle")
    s.text(gx, gy + 16, f"τ = {DEFAULT_TAU}", size=12, fill=GATE, anchor="middle", mono=True)
    s.line([(x2 + 300, gy), (gx - gr - 2, gy)], color=INK)

    # stage 3
    x3 = 1102
    s.stage_card(x3, top, 270, h, 3, "Verification")
    s.text(x3 + 20, top + 92, "Deferred words only, one", size=14, weight=600)
    s.text(x3 + 20, top + 112, "resolver per Stage-1 tag:", size=14, weight=600)
    lanes = [("EN", "UR", "lexicon prior"), ("VAR", "BOTH", "variant rule"), ("AMB", "UNK", "context decoder")]
    for i, (a, b, name) in enumerate(lanes):
        yy = top + 146 + i * 36
        s.chip(x3 + 44, yy, a, size=11, h=22, w=44)
        s.chip(x3 + 96, yy, b, size=11, h=22, w=48)
        s.text(x3 + 128, yy + 5, "→ " + name, size=13.5, fill=INK)
    s.line([(gx + gr + 2, gy), (x3 - 2, gy)], color=GATE)
    s.text((gx + gr + x3) / 2, gy - 10, "< τ", size=12.5, weight=700, fill=GATE, anchor="middle")

    # output
    xo = 1420
    s.plain_card(xo, top, 120, h, "OUTPUT", "Labels", accent=INK)
    for i, lab in enumerate(("UR", "EN", "OTH")):
        s.chip(xo + 60, top + 96 + i * 36, lab, size=12, h=26, w=60)
    s.text(xo + 20, top + 214, "+ which rule", size=12.5, fill=MUTED)
    s.text(xo + 20, top + 232, "changed it", size=12.5, fill=MUTED)
    s.line([(x3 + 270, gy), (xo - 2, gy)], color=STAGE[3][0])
    # accepted bypass
    s.line([(gx, gy - gr - 2), (gx, 146), (xo + 60, 146), (xo + 60, top - 2)], color=STAGE[2][0], sw=1.8)
    s.text(gx + 14, 140, "≥ τ  accepted: Stage 2's label is kept and pinned; Stage 3 never changes it",
           size=13, weight=600, fill=STAGE[2][0])
    s.line([(236, gy), (x1 - 2, gy)], color=INK)
    s.line([(x1 + 300, gy), (x2 - 2, gy)], color=STAGE[1][0])

    # sources under the stages
    sy = top + h + 34
    sources = [
        (x1, 300, STAGE[1][0], ["seed lexicons + variant pairs", "+ words induced from training labels"]),
        (x2, 300, STAGE[2][0], ["COMI-LINGUA Latin-script proxy (train)", "temperature T fitted on validation"]),
        (x3, 270, STAGE[3][0], ["τ and λ tuned on validation only", "fixes / breaks counted per rule"]),
    ]
    for x, w, color, lines in sources:
        s.rect(x, sy, w, 62, fill="#FFFFFF", stroke=FAINT, dash="5 4", r=10)
        for i, ln in enumerate(lines):
            s.text(x + 16, sy + 26 + i * 20, ln, size=12.5, fill=MUTED)
        s.line([(x + w / 2, sy - 2), (x + w / 2, top + h + 4)], color=color, dash="4 4", sw=1.4)

    # trace
    ty = 590
    s.rect(60, ty, W - 120, 430, shadow=True, r=16)
    s.text(84, ty + 38, "Running example, traced word by word", size=18, weight=700)
    s.text(W - 84, ty + 38, "model-free fallback (no trained checkpoint), context penalty 1.0 · gold from "
           "data/demo/running_examples.csv", size=12, fill=FAINT, anchor="end")
    col0, step = 330, 106
    rows_y = {"word": ty + 92, "tag": ty + 144, "s2": ty + 210, "final": ty + 290, "gold": ty + 352}
    s.text(84, rows_y["word"] + 5, "word", size=13, weight=700, fill=MUTED, ls=0.6)
    s.text(84, rows_y["tag"] + 5, "Stage 1 tag", size=13, weight=700, fill=STAGE[1][0])
    s.text(84, rows_y["s2"] + 0, "Stage 2 label", size=13, weight=700, fill=STAGE[2][0])
    s.text(84, rows_y["s2"] + 18, "· confidence", size=12.5, fill=STAGE[2][0])
    s.text(84, rows_y["final"] + 0, "After Stage 3", size=13, weight=700, fill=STAGE[3][0])
    s.text(84, rows_y["gold"] + 5, "gold", size=13, weight=700, fill=MUTED, ls=0.6)
    # tau band between rows
    s.add(f'<line x1="{col0 - 60}" y1="{ty + 248}" x2="{W - 84}" y2="{ty + 248}" stroke="{GATE}" '
          f'stroke-width="1" stroke-dasharray="2 4"/>')
    s.text(W - 84, ty + 243, f"gate τ = {DEFAULT_TAU}: dashed chips are deferred to Stage 3", size=11.5,
           fill=GATE, anchor="end")
    s2_ok = final_ok = 0
    for i, (word, tag, reading, rec, g) in enumerate(zip(words, tags, readings, records, gold)):
        cx = col0 + i * step
        s.text(cx, rows_y["word"] + 6, word, size=16, weight=600, anchor="middle", mono=True)
        s.chip(cx, rows_y["tag"], tag, w=64, size=12)
        if reading:
            s.text(cx, rows_y["tag"] + 30, f"→ {reading}", size=11.5, fill=TAG["VAR"][2], anchor="middle", mono=True)
        s2 = rec["stage2_label"]
        s.chip(cx, rows_y["s2"], s2, label=f"{s2} {rec['confidence']:.2f}".replace(" 0.", " ."), w=86, size=12,
               dashed=rec["deferred"], faded=rec["deferred"])
        changed = rec["label"] != s2
        s.chip(cx, rows_y["final"], rec["label"], w=64, size=12, ring=STAGE[3][0] if changed else None)
        if changed:
            s.text(cx, rows_y["final"] + 32, rec["rules"][-1], size=11, fill=STAGE[3][0], anchor="middle",
                   weight=600)
        ok_final = rec["label"] == g
        s.text(cx, rows_y["gold"] + 6, g + (" ✓" if ok_final else " ✗"), size=13, weight=600,
               fill=TAG[g][2] if g in TAG else MUTED, anchor="middle")
        s2_ok += s2 == g
        final_ok += ok_final
    n = len(words)
    for key, value, color in (("s2", s2_ok, STAGE[2][0]), ("final", final_ok, STAGE[3][0])):
        s.text(W - 84, rows_y[key] + 6, f"{value}/{n}", size=17, weight=700, fill=color, anchor="end")
    # legend
    ly = ty + 398
    cx = 96
    for tag, meaning in TAG_MEANING.items():
        w = s.chip(cx + 22, ly, tag, w=44 if tag != "BOTH" else 52, h=22, size=11)
        s.text(cx + 22 + w / 2 + 8, ly + 4.5, meaning, size=12, fill=MUTED)
        cx += w + 8 + len(meaning) * 6.3 + 30
    return s.render()


# ---- stage 1 ----------------------------------------------------------------------------------
def stage1() -> str:
    n_en, n_ur, n_pairs = lexicon_counts()
    words, tags, readings, records, gold = running_example()
    s = Svg(1000, "RU-EN-LID Stage 1: candidate expansion",
            "Each word is looked up in an English lexicon, a Roman Urdu lexicon and a list of attested spelling "
            "variants; a fixed decision order assigns one of six evidence tags and an Urdu reading. The word itself "
            "is never rewritten.")
    s.header("Stage 1 · Candidate expansion",
             "Each word is looked up, tagged, and given its Urdu reading. Evidence is added; the word itself is "
             "never rewritten.", "stage 1")
    accent, light = STAGE[1]

    # resources column
    x, w = 60, 360
    s.text(x, 160, "RESOURCES", size=12, weight=700, fill=accent, ls=1.6)
    cards = [
        ("English lexicon", f"{n_en}", "seed words",
         ["false friends removed on load:", "hai · ko · ki · se · nahi · …"]),
        ("Roman Urdu lexicon", f"{n_ur}", "seed words", ["spelt as people type it,", "not as a dictionary writes it"]),
        ("Attested variant pairs", f"{n_pairs}", "pairs",
         ["bht → bohat   nhi → nahi", "he → hai     bar → baar"]),
    ]
    y = 176
    centers = []
    for title, big, unit, lines in cards:
        centers.append(y + 59)
        s.rect(x, y, w, 118, shadow=True, r=14)
        s.text(x + 20, y + 34, title, size=16, weight=700)
        s.text(x + w - 20, y + 42, big, size=34, weight=700, fill=accent, anchor="end")
        s.text(x + w - 20, y + 62, unit, size=12, fill=MUTED, anchor="end")
        for i, ln in enumerate(lines):
            s.text(x + 20, y + 70 + i * 22, ln, size=13, fill=MUTED, mono="→" in ln)
        y += 132
    # induced
    s.rect(x, y, w, 150, fill="#FFFFFF", stroke=accent, dash="6 4", r=14)
    s.text(x + 20, y + 30, "+ induced from training labels", size=15, weight=700, fill=accent)
    s.text(x + 20, y + 50, "scripts/build_lexicon.py (optional)", size=12, fill=FAINT, mono=True)
    for i, ln in enumerate(["seen ≥ 3 times in the training split", "≥ 90 % one language  →  EN or UR list",
                            "each language ≥ 20 %  →  both lists (AMB)", "seed files are never edited"]):
        s.text(x + 20, y + 76 + i * 19, ln, size=13, fill=MUTED)
    s.line([(x + w + 4, y + 75), (x + w + 30, y + 75), (x + w + 30, centers[0]), (x + w + 6, centers[0])],
           color=accent, dash="4 4", sw=1.3)
    s.line([(x + w + 30, centers[1]), (x + w + 6, centers[1])], color=accent, dash="4 4", sw=1.3)

    # decision table
    tx, tw = 470, 680
    s.text(tx, 160, "DECISION · checked top to bottom, first match wins", size=12, weight=700, fill=accent, ls=1.6)
    s.rect(tx, 176, tw, 520, shadow=True, r=14)
    head_y = 210
    cols = [("attested", tx + 74), ("in English", tx + 166), ("in Urdu", tx + 254), ("tag", tx + 344),
            ("example", tx + 448), ("Urdu reading", tx + 590)]
    s.add(f'<rect x="{tx}" y="176" width="{tw}" height="52" rx="14" fill="{light}"/>')
    s.add(f'<rect x="{tx}" y="210" width="{tw}" height="18" fill="{light}"/>')
    for label, cxx in cols:
        s.text(cxx, 207, "variant" if label == "attested" else label, size=12.5, weight=700, fill=accent,
               anchor="middle")
    rows = [
        ("✓", "✓", "·", "BOTH", "he", "hai", "a variant that is also an English word"),
        ("·", "✓", "✓", "AMB", "main", "—", "two readings; only context can decide"),
        ("✓", "·", "·", "VAR", "bht", "bohat", "non-standard spelling of an Urdu word"),
        ("—", "✓", "—", "EN", "upset", "—", "English only"),
        ("—", "—", "✓", "UR", "tha", "—", "Urdu only"),
        ("—", "—", "—", "UNK", "jio", "—", "no lexicon evidence"),
    ]
    for i, (v, e, u, tag, ex, rd, note) in enumerate(rows):
        ry = 262 + i * 72
        if i:
            s.add(f'<line x1="{tx + 20}" y1="{ry - 36}" x2="{tx + tw - 20}" y2="{ry - 36}" stroke="{HAIR}"/>')
        for mark, cxx in ((v, cols[0][1]), (e, cols[1][1]), (u, cols[2][1])):
            color = TAG["UR"][2] if mark == "✓" else (FAINT if mark == "·" else "#B9BEC6")
            s.text(cxx, ry + 6, mark, size=17, weight=700, fill=color, anchor="middle")
        s.chip(cols[3][1], ry, tag, w=64, size=12.5)
        s.text(cols[4][1], ry + 6, ex, size=16, weight=600, anchor="middle", mono=True)
        s.text(cols[5][1], ry + 6, rd, size=15, weight=600 if rd != "—" else 400,
               fill=TAG["VAR"][2] if rd != "—" else "#B9BEC6", anchor="middle", mono=True)
        s.text(cols[4][1] - 50, ry + 26, note, size=11.5, fill=FAINT)
    s.text(tx + 20, 684, "✓ yes   — no   · either", size=12, fill=FAINT)

    # handoff column
    hx, hw = 1200, 340
    s.text(hx, 160, "HANDOFF TO STAGE 2", size=12, weight=700, fill=accent, ls=1.6)
    for i, (word, tag, rd, line2) in enumerate([("bht", "VAR", "bohat", "Urdu reading added"),
                                                 ("main", "AMB", None, "no reading: context decides")]):
        y = 176 + i * 176
        s.rect(hx, y, hw, 156, shadow=True, r=14)
        s.text(hx + 20, y + 32, "surface", size=12, fill=FAINT)
        s.text(hx + 110, y + 32, word, size=18, weight=700, mono=True)
        s.text(hx + 20, y + 68, "tag", size=12, fill=FAINT)
        s.chip(hx + 142, y + 63, tag, w=64, size=12.5)
        s.text(hx + 20, y + 102, "reading", size=12, fill=FAINT)
        s.text(hx + 110, y + 106, rd or "—", size=17, weight=600, mono=True,
               fill=TAG["VAR"][2] if rd else "#B9BEC6")
        s.text(hx + 20, y + 138, line2, size=12.5, fill=MUTED, italic=True)
    s.rect(hx, 528, hw, 168, fill=light, stroke=accent, sw=1.2, r=14)
    s.text(hx + 20, 560, "Tags are evidence, not answers.", size=15, weight=700, fill=accent)
    for i, ln in enumerate(["Stage 2 sees the tag and reading", "next to the original word and", "decides in context. AMB, VAR,",
                            "BOTH and UNK never appear as", "output labels."]):
        s.text(hx + 20, 588 + i * 20, ln, size=13.5, fill=INK)
    s.line([(tx + tw + 4, 262), (hx - 6, 262)], color=accent)

    # sentence strip
    sy = 740
    s.rect(60, sy, W - 120, 236, shadow=True, r=16)
    s.text(84, sy + 36, "What Stage 1 hands on for the running example", size=17, weight=700)
    s.text(W - 84, sy + 36, "computed by src/classifier/features.py", size=12, fill=FAINT, anchor="end", mono=True)
    col0, step = 150, 128
    for i, (word, tag, rd) in enumerate(zip(words, tags, readings)):
        cx = col0 + i * step
        s.text(cx, sy + 94, word, size=18, weight=600, anchor="middle", mono=True)
        s.line([(cx, sy + 106), (cx, sy + 128)], color=FAINT, sw=1.2)
        s.chip(cx, sy + 146, tag, w=70, size=12.5)
        s.text(cx, sy + 190, rd or "·", size=14, weight=600 if rd else 400,
               fill=TAG["VAR"][2] if rd else "#C3C7CE", anchor="middle", mono=True)
    s.text(84 - 2, sy + 98, "word", size=12, fill=FAINT)
    s.text(84 - 2, sy + 150, "tag", size=12, fill=FAINT)
    s.text(84 - 2, sy + 194, "reading", size=12, fill=FAINT)
    return s.render()


# ---- stage 2 ----------------------------------------------------------------------------------
def vector(s: Svg, x, y, cells, color, strength, seed, label, sub, cell=15, gap=3):
    """A row of embedding cells; strength scales opacity (0 = a freshly zero-initialised vector)."""
    for i in range(cells):
        v = (math.sin(seed * 12.9898 + i * 78.233) * 43758.5453) % 1
        alpha = 0.10 + 0.85 * abs(v) * strength
        s.add(f'<rect x="{x + i * (cell + gap)}" y="{y}" width="{cell}" height="{cell + 7}" rx="3" '
              f'fill="{color}" fill-opacity="{alpha:.2f}" stroke="{color}" stroke-opacity="0.35"/>')
    end = x + cells * (cell + gap)
    s.text(end + 14, y + 10, label, size=14, weight=700)
    s.text(end + 14, y + 27, sub, size=12, fill=MUTED)


def stage2() -> str:
    s = Svg(900, "RU-EN-LID Stage 2: candidate-conditioned XLM-R",
            "Stage 1's tag and Urdu reading are added to each word piece's input embedding; with the new "
            "parameters at zero the model equals plain XLM-R. Twelve transformer layers read the sentence and a "
            "token head outputs UR, EN or OTH with a temperature-scaled confidence.")
    s.header("Stage 2 · Candidate-conditioned XLM-R",
             "Stage 1's evidence is added to each word's input embedding, so all 12 layers can weigh the lexicon "
             "against the sentence.", "stage 2")
    accent, light = STAGE[2]
    var = TAG["VAR"][1]

    # embedding sum
    x0, y0 = 60, 150
    s.rect(x0, y0, 620, 470, shadow=True, r=16)
    s.text(x0 + 24, y0 + 36, "Input embedding of one word piece", size=17, weight=700)
    s.text(x0 + 24, y0 + 58, "example: bht  (tag VAR, reading bohat)", size=13, fill=MUTED, mono=True)
    vx = x0 + 40
    vector(s, vx, y0 + 92, 14, INK, 1.0, 1, "word-piece embedding", "XLM-R vocabulary, pretrained")
    s.text(vx - 22, y0 + 160, "+", size=24, weight=700, fill=MUTED)
    vector(s, vx, y0 + 146, 14, var, 0.55, 2, "tag embedding · VAR", "7 tags × 768, new, starts at 0")
    s.text(vx - 22, y0 + 214, "+", size=24, weight=700, fill=MUTED)
    s.text(vx - 4, y0 + 270 - 56 + 2, "", size=1)
    vector(s, vx, y0 + 200, 14, var, 0.35, 3, "g × mean(reading pieces)", "'bohat', ≤ 4 pieces · gate g starts at 0")
    s.add(f'<line x1="{vx}" y1="{y0 + 244}" x2="{vx + 14 * 18 - 3}" y2="{y0 + 244}" stroke="{INK}" '
          f'stroke-width="1.6"/>')
    vector(s, vx, y0 + 258, 14, accent, 1.0, 4, "e(piece)", "+ position · LayerNorm → layers")
    # zero-init explainer
    s.rect(x0 + 24, y0 + 318, 572, 128, fill=light, stroke=accent, sw=1.1, r=12)
    s.text(x0 + 44, y0 + 346, "Safe start", size=15, weight=700, fill=accent)
    for i, ln in enumerate(["Tag table and gate start at zero, so at step 0 the model is", "exactly the plain XLM-R "
                            "classifier (tested to 1e-6). Any later", "difference is learned from data, and the "
                            "ablation changes", "nothing else."]):
        s.text(x0 + 44, y0 + 370 + i * 19, ln, size=13.5)

    # encoder stack
    ex, ey = 740, 150
    s.rect(ex, ey, 400, 470, shadow=True, r=16)
    s.text(ex + 24, ey + 36, "Encoder", size=17, weight=700)
    s.text(ex + 24, ey + 58, "xlm-roberta-base · 12 layers · 768-d", size=13, fill=MUTED, mono=True)
    for i in range(6):
        ly = ey + 108 + i * 34
        s.rect(ex + 46 + i * 4, ly, 300 - i * 8, 26, fill=accent if i == 0 else light, stroke=accent, sw=1, r=7,
               opacity=None)
        s.text(ex + 196, ly + 18, "Transformer layer" if i else "Transformer layer 12", size=12.5,
               weight=600, fill="#FFFFFF" if i == 0 else accent, anchor="middle")
    s.text(ex + 196, ey + 330, "⋮   × 12", size=16, weight=700, fill=accent, anchor="middle")
    # sentence tokens into the stack
    toks = [("box", "EN"), ("band", "AMB"), ("nhi", "VAR"), ("tha", "UR"), ("main", "AMB")]
    for i, (word, tag) in enumerate(toks):
        cx = ex + 68 + i * 66
        s.chip(cx, ey + 380, tag, label=word, w=58, size=12)
        s.line([(cx, ey + 366), (cx, ey + 346)], color=FAINT, sw=1.2)
    s.text(ex + 196, ey + 424, "every word attends to the whole sentence;", size=12.5, fill=MUTED, anchor="middle")
    s.text(ex + 196, ey + 442, "chip colour = Stage-1 tag fed in", size=12.5, fill=MUTED, anchor="middle")
    s.line([(x0 + 620 + 4, y0 + 268), (ex - 4, y0 + 268)], color=accent)

    # head and output
    hx = 1200
    s.rect(hx, 150, 340, 470, shadow=True, r=16)
    s.text(hx + 24, 186, "Head and confidence", size=17, weight=700)
    steps = [("token head", "linear, 768 → 3 logits z per piece"), ("softmax(z / T)", "T fitted on validation NLL"),
             ("first piece of each word", "one prediction per word")]
    for i, (a, b) in enumerate(steps):
        yy = 222 + i * 70
        s.rect(hx + 24, yy, 292, 52, fill=light, stroke=accent, sw=1, r=10)
        s.text(hx + 40, yy + 22, a, size=14, weight=700, fill=accent, mono="(" in a)
        s.text(hx + 40, yy + 41, b, size=12.5, fill=MUTED)
        if i:
            s.line([(hx + 170, yy - 16), (hx + 170, yy - 2)], color=accent, sw=1.3)
    s.text(hx + 24, 452, "Output contract", size=14, weight=700)
    for i, lab in enumerate(("UR", "EN", "OTH")):
        s.chip(hx + 54 + i * 64, 484, lab, w=54, size=12)
    s.text(hx + 24, 526, "label = argmax, confidence = max p", size=13, fill=INK, mono=True)
    s.text(hx + 24, 552, "Tags go in; only UR / EN / OTH", size=13, fill=MUTED)
    s.text(hx + 24, 571, "come out. The confidence decides", size=13, fill=MUTED)
    s.text(hx + 24, 590, "what Stage 3 may touch.", size=13, fill=MUTED)
    s.line([(ex + 400 + 4, y0 + 268), (hx - 4, y0 + 268)], color=accent)

    # training facts
    fy = 660
    s.text(60, fy, "TRAINING · src/classifier/train.py --candidates none | tag | tag+reading", size=12, weight=700,
           fill=accent, ls=1.2)
    facts = [
        ("Tag dropout 15 %", ["per word, the tag becomes UNK", "and the reading is hidden, so", "the model cannot lean on a",
                              "lexicon that is mostly UNK"]),
        ("Two learning rates", ["encoder at the usual rate", "(3e-5 in the iteration-1 run);", "new tag and gate params",
                                "at 1e-3 (--cand-lr)"]),
        ("Three ablation arms", ["none · tag · tag+reading", "same data, seeds and", "hyperparameters; only the",
                                 "input evidence differs"]),
        ("Labels", ["first piece of a word carries", "its label; other pieces and", "unresolved (IGN) words are",
                    "skipped by the loss"]),
        ("Calibration", ["one temperature T, fitted", "after training on validation", "NLL; labels never change,",
                         "re-tune τ afterwards"]),
    ]
    cw, gap = 280, 15
    for i, (title, lines) in enumerate(facts):
        x = 60 + i * (cw + gap) - (0 if i < 4 else 0)
        w = cw if i < 4 else W - 60 - x
        s.rect(x, fy + 16, w, 178, shadow=True, r=14)
        s.add(f'<rect x="{x}" y="{fy + 16}" width="5" height="178" rx="2" fill="{accent}"/>')
        s.text(x + 22, fy + 50, title, size=15, weight=700)
        for j, ln in enumerate(lines):
            s.text(x + 22, fy + 80 + j * 22, ln, size=13, fill=MUTED)
    return s.render()


# ---- stage 3 ----------------------------------------------------------------------------------
def stage3() -> str:
    s = Svg(920, "RU-EN-LID Stage 3: gate and verification",
            "A confidence gate keeps words Stage 2 is sure about. Deferred words go to one resolver chosen by "
            "their Stage-1 tag: lexicon priors, variant consistency, or a context decoder that charges a penalty "
            "for each Urdu-English switch.")
    s.header("Stage 3 · Gate and verification",
             "Confident words are kept. Each deferred word goes to the one resolver that has evidence for it.",
             "stage 3")
    accent, light = STAGE[3]

    # gate
    s.plain_card(60, 166, 220, 172, "FROM STAGE 2", "label + conf", accent=STAGE[2][0])
    s.text(80, 242, "per word, e.g.", size=13, fill=MUTED)
    s.chip(136, 274, "EN", label="main EN .30", w=112, size=12, dashed=True, faded=True)
    s.chip(136, 310, "UR", label="tha UR .90", w=112, size=12)
    gx, gy, gr = 380, 251, 56
    s.add(f'<path d="M {gx} {gy - gr} L {gx + gr} {gy} L {gx} {gy + gr} L {gx - gr} {gy} z" fill="{GATE_L}" '
          f'stroke="{GATE}" stroke-width="1.6" filter="url(#shadow)"/>')
    s.text(gx, gy - 4, "conf ≥ τ ?", size=14, weight=700, fill=GATE, anchor="middle")
    s.text(gx, gy + 16, f"τ = {DEFAULT_TAU}", size=12, fill=GATE, anchor="middle", mono=True)
    s.line([(280, gy), (gx - gr - 2, gy)], color=INK)
    # accepted
    s.rect(60, 386, 360, 106, fill=STAGE[2][1], stroke=STAGE[2][0], sw=1.2, r=14)
    s.text(80, 418, "≥ τ  Accepted", size=16, weight=700, fill=STAGE[2][0])
    s.text(80, 444, "pinned: Stage 2's label is final.", size=13.5)
    s.text(80, 466, "No rule and no neighbour can change it.", size=13.5)
    s.line([(gx, gy + gr + 2), (gx, 382)], color=STAGE[2][0])

    # lanes
    lx = 520
    s.text(lx, 160, "< τ  DEFERRED · routed by the Stage-1 tag", size=12, weight=700, fill=GATE, ls=1.4)
    s.line([(gx + gr + 2, gy), (lx - 4, gy)], color=GATE)
    lanes = [
        (["EN"], "Stage 2 said UR", "english-lexicon-prior", "EN"),
        (["UR"], "Stage 2 said EN", "urdu-lexicon-prior", "UR"),
        (["VAR", "BOTH"], "Stage 2 said EN", "variant-consistency", "UR"),
        (["AMB", "UNK"], "no lexicon evidence", "context decoder", "from neighbours"),
    ]
    for i, (tagset, cond, rule, result) in enumerate(lanes):
        y = 176 + i * 96
        is_ctx = rule == "context decoder"
        s.rect(lx, y, 470, 80, fill=CARD if not is_ctx else light, stroke=accent if is_ctx else HAIR,
               sw=1.4 if is_ctx else 1, r=14, shadow=not is_ctx)
        cx = lx + 40
        for t in tagset:
            w = s.chip(cx, y + 28, t, w=50 if t != "BOTH" else 56, size=12)
            cx += w + 6
        s.text(lx + 20, y + 62, cond, size=12.5, fill=MUTED)
        s.line([(lx + 196, y + 40), (lx + 222, y + 40)], color=accent, sw=1.4)
        s.text(lx + 232, y + 36, rule, size=14, weight=700, fill=accent, mono=True)
        if result in ("EN", "UR"):
            s.text(lx + 232, y + 60, "→", size=14, fill=MUTED)
            s.chip(lx + 274, y + 56, result, w=46, h=22, size=11.5)
        else:
            s.text(lx + 232, y + 60, "→ UR / EN / OTH, from neighbours", size=12.5, fill=MUTED)
        if i:
            s.line([(lx - 14, gy), (lx - 14, y + 40), (lx - 2, y + 40)], color=GATE, sw=1.3, arrow=True)
    s.text(lx, 576, "Otherwise Stage 2's label is kept. One resolver per tag; ambiguous words", size=12.5,
           fill=MUTED)
    s.text(lx, 594, "go only to context because both lexicons accept them.", size=12.5, fill=MUTED)

    # context decoder lattice
    cx0, cy0, cw = 1040, 150, 500
    s.rect(cx0, cy0, cw, 470, shadow=True, r=16)
    s.text(cx0 + 24, cy0 + 36, "Context decoder", size=17, weight=700, fill=accent)
    s.text(cx0 + 24, cy0 + 58, "two-state Viterbi over the current language", size=13, fill=MUTED)
    s.rect(cx0 + 24, cy0 + 76, cw - 48, 66, fill=light, r=10, stroke=accent, sw=1)
    s.rich(cx0 + 40, cy0 + 104, [("cost = Σ −log p₂(yᵢ) ", {"mono": True, "weight": 600}),
                                 ("+ λ · #switches", {"mono": True, "weight": 700, "fill": accent})], size=14)
    s.text(cx0 + 40, cy0 + 128, "OTH keeps the language running · λ = 0 → off", size=12.5, fill=MUTED)
    cols = [("tha", "pinned", {"UR": None}), ("main", "deferred · AMB", {"UR": 0.5, "EN": 0.5}),
            ("bht", "pinned", {"UR": None})]
    rows = {"UR": cy0 + 250, "EN": cy0 + 340}
    xs = [cx0 + 100, cx0 + 260, cx0 + 420]
    s.text(cx0 + 30, rows["UR"] + 5, "UR", size=13, weight=700, fill=TAG["UR"][2])
    s.text(cx0 + 30, rows["EN"] + 5, "EN", size=13, weight=700, fill=TAG["EN"][2])
    # edges first
    s.line([(xs[0] + 22, rows["UR"]), (xs[1] - 24, rows["UR"])], color=accent, sw=3, arrow=False)
    s.line([(xs[1] + 22, rows["UR"]), (xs[2] - 24, rows["UR"])], color=accent, sw=3, arrow=False)
    s.line([(xs[0] + 20, rows["UR"] + 10), (xs[1] - 22, rows["EN"] - 10)], color="#C3C7CE", sw=1.4, arrow=False,
           dash="5 4")
    s.line([(xs[1] + 20, rows["EN"] - 10), (xs[2] - 22, rows["UR"] + 10)], color="#C3C7CE", sw=1.4, arrow=False,
           dash="5 4")
    s.text((xs[0] + xs[1]) / 2 - 8, rows["UR"] + 64, "+λ", size=13, weight=700, fill=FAINT, anchor="middle")
    s.text((xs[1] + xs[2]) / 2 + 8, rows["UR"] + 64, "+λ", size=13, weight=700, fill=FAINT, anchor="middle")
    for x, (word, note, opts) in zip(xs, cols):
        s.text(x, cy0 + 186, word, size=16, weight=700, mono=True, anchor="middle")
        s.text(x, cy0 + 206, note, size=11.5, fill=FAINT, anchor="middle")
        for lab, y in rows.items():
            if lab in opts:
                best = lab == "UR"
                s.chip(x, y, lab, w=44, h=30, size=12.5, ring=accent if best else None)
                p = opts[lab]
                if p is not None:
                    s.text(x, y + 34, f"−log {p} = {-math.log(p):.2f}", size=11.5, fill=MUTED, anchor="middle",
                           mono=True)
            else:
                s.add(f'<circle cx="{x}" cy="{y}" r="4" fill="#D5D8DE"/>')
    s.text(cx0 + 24, cy0 + 414, "UR path 0.69 + 0  <  EN path 0.69 + 2λ", size=13.5, weight=700, mono=True)
    s.text(cx0 + 24, cy0 + 438, "so main → UR whenever λ > 0 (it is Urdu here).", size=13, fill=MUTED)
    s.text(cx0 + 24, cy0 + 456, "p₂ shown: model-free fallback; a trained model differs.", size=11.5, fill=FAINT)

    # bottom callouts
    by = 660
    s.rect(60, by, 720, 180, shadow=True, r=16)
    s.add(f'<rect x="60" y="{by}" width="5" height="180" rx="2" fill="{accent}"/>')
    s.text(84, by + 36, "Why variants are not re-decoded", size=16, weight=700)
    s.rich(84, by + 70, [("same issue ", {"mono": True, "weight": 600, "fill": TAG["EN"][2]}),
                         ("he", {"mono": True, "weight": 700, "fill": TAG["VAR"][2]}),
                         ("  (he = hai)", {"fill": MUTED})], size=16)
    for i, ln in enumerate(["With context on, he followed its English neighbours and flipped to EN.",
                            "English nouns inside an Urdu sentence are common, so a switch penalty is the",
                            "wrong prior for variants. They keep the variant rule (an Urdu prior)."]):
        s.text(84, by + 104 + i * 22, ln, size=13.5, fill=INK if i == 0 else MUTED)
    s.rect(810, by, 730, 180, shadow=True, r=16)
    s.add(f'<rect x="810" y="{by}" width="5" height="180" rx="2" fill="{GATE}"/>')
    s.text(834, by + 36, "Tuned and measured, not assumed", size=16, weight=700)
    for i, (a, b) in enumerate([("scripts/tune_stage3.py", "grid over τ × λ on validation only"),
                                ("src/eval/selective.py", "coverage · accepted vs deferred accuracy"),
                                ("", "fixes and breaks per rule and per Stage-1 tag")]):
        s.text(834, by + 72 + i * 30, a, size=13, mono=True, weight=600, fill=INK)
        s.text(1060, by + 72 + i * 30, b, size=13.5, fill=MUTED)
    s.text(834, by + 160, "If context breaks more than it fixes on validation, tuning picks λ = 0.", size=13,
           fill=GATE, weight=600)
    s.text(60, by + 230, "Rule conditions as in src/verification/rules.py (apply_rules, apply_context); "
           "CONTEXT_CATEGORIES = ambiguous, unknown.", size=12, fill=FAINT)
    return s.render()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT), help="output folder (default: docs/architecture)")
    out = Path(parser.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    for name, build in (("overview", overview), ("stage1", stage1), ("stage2", stage2), ("stage3", stage3)):
        (out / f"{name}.svg").write_text(build(), encoding="utf-8")
        print(f"[written] {out / (name + '.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
