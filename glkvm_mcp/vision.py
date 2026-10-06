# OCR element tree for the GLKVM target screen.
#
# RapidOCR produces line-level text with 4-point boxes. We cluster lines into
# blocks (paragraph-like stacks), and estimate word/letter bboxes inside a line
# by proportional character-width accumulation along the line box. Coordinates
# are target-screen pixels (same frame as /streamer/snapshot).

from __future__ import annotations

import time
from typing import Any

import numpy as np

_CHAR_W = {
    "upper": 0.72,
    "lower": 0.55,
    "digit": 0.62,
    "wide": 1.0,     # CJK / fullwidth
    "punct": 0.34,
    "space": 0.0,
}


def _char_w(ch: str) -> float:
    if "⺀" <= ch <= "鿿" or "＀" <= ch <= "￯" or ch in ("—", "–", "…", "·"):
        return _CHAR_W["wide"]
    if ch == " ":
        return _CHAR_W["space"]
    if ch.isupper():
        return _CHAR_W["upper"]
    if ch.isdigit():
        return _CHAR_W["digit"]
    if ch.isalpha():
        return _CHAR_W["lower"]
    return _CHAR_W["punct"]


def _split_positions(text: str) -> list[tuple[int, int]]:
    """Word spans (start, end) split on whitespace."""
    words, start = [], 0
    in_word = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if in_word:
                words.append((start, i))
            in_word = False
        else:
            if not in_word:
                start = i
                in_word = True
    if in_word:
        words.append((start, len(text)))
    return words


def _sub_bbox(x0: float, y0: float, x1: float, y1: float, text: str,
              s: int, e: int) -> tuple[float, float, float, float]:
    """Estimated bbox for text[s:e] inside the line box, proportional char widths."""
    cum = [0.0]
    for ch in text:
        cum.append(cum[-1] + _char_w(ch))
    total = cum[-1] or 1.0
    left = cum[s] / total
    right = cum[e] / total
    w = (x1 - x0)
    bx0 = x0 + left * w
    bx1 = x0 + right * w
    if bx1 - bx0 < 2:
        bx0 -= 1.0
        bx1 += 1.0
    return (bx0, y0, bx1, y1)


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


class Engine:
    def __init__(self) -> None:
        self._ocr = None

    def _engine(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            # threads=8 measured fastest on the 32-core host (default threading oversubscribes)
            try:
                self._ocr = RapidOCR(
                    det_limit_side_len=1280, det_limit_side_len_mode="max",
                    det_thresh=0.3, det_box_thresh=0.5, det_unclip_ratio=1.7,
                    intra_op_num_threads=8,
                )
            except Exception:
                self._ocr = RapidOCR()
        return self._ocr

    def run(self, img, scale: float = 1.0) -> tuple[list, float]:
        """img: PIL Image (full res). Returns (result, ms). result entries: [quad, text, score]."""
        eng = self._engine()
        pil = img
        fx = 1.0 / scale if scale and scale > 0 else 1.0
        if scale and scale < 1.0:
            pil = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        arr = np.asarray(pil)[:, :, ::-1]  # RGB -> BGR for cv2-based pipeline
        t0 = time.perf_counter()
        res, _ = eng(arr)
        ms = (time.perf_counter() - t0) * 1000
        out = []
        if res:
            for quad, text, score in res:
                q = [[float(p[0]) * fx, float(p[1]) * fx] for p in quad]
                out.append([q, str(text), float(score)])
        return (out, ms)


def quad_to_bbox(quad: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def build_tree(res: list, w: int, h: int, sha: str, ts: float, ocr_ms: float,
               letters: bool = False, min_score: float = 0.45) -> dict:
    """OCR results -> blocks/lines/words(/letters) tree with screen-px bboxes."""
    lines = []
    for i, (quad, text, score) in enumerate(res):
        if score < min_score or not text.strip():
            continue
        x0, y0, x1, y1 = quad_to_bbox(quad)
        lines.append({
            "i": len(lines),
            "text": text,
            "score": round(score, 3),
            "bbox": [round(x0), round(y0), round(x1 - x0), round(y1 - y0)],
            "cx": (x0 + x1) / 2,
            "cy": (y0 + y1) / 2,
        })
    # sort by vertical center, then x
    lines.sort(key=lambda l: (round(l["cy"] / 8), l["bbox"][0]))
    # cluster into blocks: consecutive lines with small vertical gap and near x-overlap
    blocks: list[list[dict]] = []
    for line in lines:
        placed = False
        if blocks:
            last = blocks[-1]
            pl = last[-1]
            lh = max(pl["bbox"][3], line["bbox"][3], 10)
            l0, l1 = pl["bbox"][1], pl["bbox"][1] + pl["bbox"][3]
            c0, c1 = line["bbox"][1], line["bbox"][1] + line["bbox"][3]
            same_block = (min(l1, c1) - max(l0, c0) > -0.45 * lh)
            # also treat horizontally-near same-row text as one line chunk
            if same_block:
                last.append(line)
                placed = True
                continue
            # start new block otherwise
        if not placed:
            blocks.append([line])
    out_blocks = []
    for bi, blk in enumerate(blocks):
        blk.sort(key=lambda l: (round(l["cy"] / 8), l["bbox"][0]))
        bl = {"id": f"b{bi}", "lines": []}
        for li, line in enumerate(blk):
            x0, y0, w_, h_ = line["bbox"]
            x1 = x0 + w_
            y1b = y0 + h_
            words = []
            for wi, (s, e) in enumerate(_split_positions(line["text"])):
                wx0, wy0, wx1, wy1 = _sub_bbox(x0, y0, x1, y1b, line["text"], s, e)
                wnode = {
                    "id": f"b{bi}l{li}w{wi}",
                    "text": line["text"][s:e],
                    "bbox": [round(wx0), round(wy0), round(wx1 - wx0), round(wy1 - wy0)],
                }
                if letters and (e - s) <= 18:
                    chars = []
                    total_w = sum(_char_w(c) for c in line["text"]) or 1.0
                    acc = sum(_char_w(c) for c in line["text"][:s])
                    for k in range(s, e):
                        cw = _char_w(line["text"][k])
                        cx0 = x0 + acc / total_w * (x1 - x0)
                        acc2 = acc + cw
                        cx1 = x0 + acc2 / total_w * (x1 - x0)
                        chars.append({
                            "id": f"b{bi}l{li}w{wi}c{k - s}",
                            "text": line["text"][k],
                            "bbox": [round(cx0), round(wy0), round(max(1.0, cx1 - cx0)), round(wy1 - wy0)],
                        })
                        acc = acc2
                    if chars:
                        wnode["letters"] = chars
                if wnode["bbox"][2] >= 2:
                    words.append(wnode)
            ln = {
                "id": f"b{bi}l{li}",
                "text": line["text"],
                "score": line["score"],
                "bbox": line["bbox"],
            }
            if words:
                ln["words"] = words
            bl["lines"].append(ln)
        out_blocks.append(bl)
    return {
        "frame": {"w": w, "h": h, "sha12": sha, "ocr_ts": round(ts, 3), "ocr_ms": round(ocr_ms, 1)},
        "counts": {"blocks": len(out_blocks), "lines": sum(len(b["lines"]) for b in out_blocks)},
        "blocks": out_blocks,
    }


def flatten(tree: dict) -> list[dict]:
    """All nodes as flat click targets: words (preferred) plus lines."""
    nodes = []
    for b in tree["blocks"]:
        for l in b["lines"]:
            bx0, by0, bw, bh = l["bbox"]
            nodes.append({"level": "line", "id": l["id"], "text": l["text"],
                          "bbox": l["bbox"],
                          "center": [bx0 + bw / 2, by0 + bh / 2]})
            for wd in l.get("words", []):
                wx0, wy0, ww, wh = wd["bbox"]
                nodes.append({"level": "word", "id": wd["id"], "text": wd["text"],
                              "bbox": wd["bbox"],
                              "center": [wx0 + ww / 2, wy0 + wh / 2]})
    return nodes


def find_text(tree: dict, query: str, nth: int = 0, exact: bool = False) -> list[dict]:
    q = query.casefold().strip()
    nodes = flatten(tree)
    matches = []
    for n in nodes:
        t = n["text"].casefold().strip()
        if (exact and t == q) or (not exact and q in t):
            matches.append(n)
    # prefer word-level hits and earlier screen position
    matches.sort(key=lambda m: (m["level"] != "word", m["center"][1], m["center"][0]))
    if nth < 0:
        return matches[nth:] if nth == -1 else matches
    return matches[nth:]


def crop_for(img, node: dict, pad: int = 24):
    x0, y0, bw, bh = node["bbox"]
    cx0, cy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    cx1 = min(img.width, int(x0 + bw) + pad)
    cy1 = min(img.height, int(y0 + bh) + pad)
    return img.crop((cx0, cy0, cx1, cy1))