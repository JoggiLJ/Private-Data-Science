"""card_grid_ocr.py — deterministic OpenCV reader for "suit symbol in a
gridded panel" IQ puzzles (a ragged left-hand question grid + a right-hand
set of lettered answer panels).

Pipeline
--------
1. Panels are gray-filled squares with a yellow grid overlay. The overlay
   cuts the gray fill into small tiles, so a naive "mask the gray fill and
   findContours" pass fragments each panel into ~9 tiny tiles.
2. The image is split into a LEFT group (question matrix) and a RIGHT group
   (lettered answers) by finding the single largest gap in gray-fill pixel
   columns (robust to crops/margins — it doesn't assume a fixed pixel split
   point).
3. Each group is a 3x3 outer matrix with exactly one cell blank (the LEFT
   group's "?" cell, and the RIGHT group's unused 9th answer slot) — true
   for every question in this puzzle family. Individual PANELS are found by
   dividing each group's bounding box evenly into a 3x3 template grid and
   keeping only the cells with real gray-fill density.
   This deliberately does NOT use blob connectivity (findContours /
   connectedComponents) to separate panels, even with a small morphological
   close: adjacent panels in this rendering are placed edge-to-edge with no
   reliable gap between them — some pairs are separated by a real ~12-14px
   gap, others are pixel-adjacent (touching or 1px overlapping via
   anti-aliasing) — so any blob-based split either merges touching panels
   together or fragments a single panel, depending on the kernel size. The
   3x3 template grid sidesteps that entirely because the *group* bounding
   box (not each panel) is reliably measurable, and the panel layout within
   it is fixed and regular. Verified against all 25 questions in this pack:
   every one yields exactly 8/9 populated cells per group.
4. Each panel's black-ink symbols are isolated by color thresholding (the
   yellow grid lines and gray fill are far from pure black, so this cleanly
   separates the "ink" from the panel decoration), followed by a small
   morphological close (see INK_CLOSE_KERNEL) that reconnects each glyph's
   body to its stem/base where a grid line had split them apart.
5. Each symbol is classified into one of the four card suits from its
   VERTICAL WIDTH PROFILE (top-heavy heart vs. bottom-heavy spade vs.
   symmetric diamond vs. wider-than-tall club) — see the note in
   classify_suit() on why scalar contour metrics and a font-glyph template
   match were tried first and rejected.
6. Output: the LEFT grid is emitted as a nested, row-preserving structure
   (plus a pretty ASCII render) since its 2D order is meaningful; the RIGHT
   panels are emitted as a flat {letter: cell} dict in reading order, since
   only the letter identity matters, not 2D position.

Usage:
    python card_grid_ocr.py IMAGE.png
    python card_grid_ocr.py IMAGE.png --debug     # dump raw features per panel
    python card_grid_ocr.py IMAGE.png --json out.json
"""
from __future__ import annotations
import argparse
import json
import os
import string
import sys

import cv2
import numpy as np

png_files = ["question-01.png", "question-02.png", "question-03.png",
             "question-04.png", "question-05.png", "question-06.png",
             "question-07.png", "question-08.png", "question-09.png",
             "question-10.png", "question-11.png", "question-12.png",
             "question-13.png", "question-14.png", "question-15.png",
             "question-16.png", "question-17.png", "question-18.png",
             "question-19.png", "question-20.png", "question-21.png",
             "question-22.png", "question-23.png", "question-24.png",
             "question-25.png"]

# -----------------------------------------------------------------------------
# Panel detection
# -----------------------------------------------------------------------------
GRAY_LO, GRAY_HI = (200, 200, 200), (235, 235, 235)   # panel background fill
BLACK_LO, BLACK_HI = (0, 0, 0), (60, 60, 60)          # ink color

N_GRID = 3            # every group is an outer 3x3 matrix (one cell blank)
PANEL_DENSITY_MIN = 0.20   # min gray-fill fraction for a template cell to count as a real panel

# The suit glyphs are rendered with a thin gap between the body and its
# stem/base (a spade's pedestal, a club's stalk), and a yellow reference-grid
# line passing through a symbol widens that gap enough to break the shape into
# two separate blobs. RETR_EXTERNAL then sees only the (wider-than-tall) body
# and the tiny stem gets dropped by the area filter, so an intact spade reads
# as a stray "club". A small morphological close bridges those ~2-3px gaps and
# restores the whole glyph before contour finding. The 5px kernel is large
# enough to reconnect every split seen across all 25 questions, yet far smaller
# than the ~25px center-to-center spacing between neighbouring symbols, so it
# never merges two distinct symbols together.
INK_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# A real panel's gray fill is chopped by the yellow grid into tiles, but even
# the smallest tile is >=522px^2 across all 25 questions; any gray connected
# component below this is a stray speck (an anti-aliasing artifact near text
# — e.g. the "3." question label in question-03 leaves a 12px cluster that
# happens to fall in the gray color range). The largest such speck seen is
# 12px^2, so this floor sits in a very wide empty gap. It matters because a
# single stray speck above/left of the real panels inflates the GROUP bounding
# box (find_group_bboxes uses raw pixel min/max), which then shifts the evenly
# divided 3x3 template grid off the true panels and truncates a whole row —
# dropping the symbols in it (this is what made question-03 box1 read 4 of its
# 6 symbols: the bottom row got cut off the panel crop).
GRAY_SPECK_MAX = 300   # px^2; between the 12px specks and the 522px real tiles


def gray_mask_of(img: np.ndarray) -> np.ndarray:
    mask = cv2.inRange(img, GRAY_LO, GRAY_HI)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= GRAY_SPECK_MAX:
            mask[labels == i] = 0
    return mask


def ink_mask_of(img: np.ndarray) -> np.ndarray:
    """Black-ink mask with body/stem gaps closed — see INK_CLOSE_KERNEL."""
    black = cv2.inRange(img, BLACK_LO, BLACK_HI)
    return cv2.morphologyEx(black, cv2.MORPH_CLOSE, INK_CLOSE_KERNEL)


def find_group_bboxes(gray_mask: np.ndarray) -> tuple[tuple, tuple]:
    """Split all gray-fill pixels into a LEFT group (question matrix) and a
    RIGHT group (lettered answers) by the single largest gap in occupied
    pixel COLUMNS, then return each group's own tight bounding box
    (x0, y0, x1, y1). Operating on raw pixel columns (not panel boxes, which
    don't exist yet at this point) makes this robust to crops/margins and to
    panels that are touching/overlapping — it only needs ONE real gap to
    exist between the two groups, not between individual panels."""
    col_present = np.where(gray_mask.sum(axis=0) > 0)[0]
    if len(col_present) < 2:
        raise ValueError("no gray panels found in image")
    gaps = np.diff(col_present)
    split_i = int(np.argmax(gaps))
    split_x = int((col_present[split_i] + col_present[split_i + 1]) / 2)

    ys, xs = np.where(gray_mask > 0)

    def bbox(sel: np.ndarray) -> tuple:
        return (int(xs[sel].min()), int(ys[sel].min()),
                 int(xs[sel].max()), int(ys[sel].max()))

    return bbox(xs <= split_x), bbox(xs > split_x)


def panels_in_group(gray_mask: np.ndarray, bbox: tuple) -> list[tuple]:
    """Locate individual panels within one group by dividing its bounding
    box evenly into an N_GRID x N_GRID template grid and keeping only the
    cells that actually have gray fill in them (density > threshold).

    This intentionally does NOT try to separate panels via blob connectivity
    (e.g. findContours after a morphological close) — see the module
    docstring for why that's unreliable here (panels are rendered edge-to-
    edge with an inconsistent, sometimes-zero gap between them). Dividing
    the reliably-measured GROUP bbox into a fixed template grid sidesteps
    the problem, since the panel layout within a group is fixed (a 3x3
    matrix with exactly one cell blank) even though panel-to-panel spacing
    isn't.

    The template slice itself is only a rough estimate of each panel's
    extent — the group bbox doesn't divide perfectly evenly by 3, so a
    slice can carry a few px of the inter-panel gap on one edge and be a
    few px short on the opposite edge. Each returned bbox is therefore
    tightened to the actual gray-fill pixels found inside its slice (never
    grown beyond it — real content should never live outside its own
    template slice by more than a px of anti-aliasing), so panels come back
    flush against each other/the group border instead of carrying that slop.

    Returns a list of (row, col, (x, y, w, h)) for every populated cell,
    in no particular order.
    """
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    cell_w, cell_h = w / N_GRID, h / N_GRID

    panels = []
    for r in range(N_GRID):
        for c in range(N_GRID):
            px0 = int(x0 + c * cell_w)
            px1 = int(x0 + (c + 1) * cell_w)
            py0 = int(y0 + r * cell_h)
            py1 = int(y0 + (r + 1) * cell_h)
            cell = gray_mask[py0:py1, px0:px1]
            density = float((cell > 0).mean()) if cell.size else 0.0
            if density > PANEL_DENSITY_MIN:
                ys, xs = np.where(cell > 0)
                tx0, tx1 = int(xs.min()), int(xs.max()) + 1
                ty0, ty1 = int(ys.min()), int(ys.max()) + 1
                panels.append((r, c, (px0 + tx0, py0 + ty0,
                                       tx1 - tx0, ty1 - ty0)))
    return panels


def box_number(row: int, col: int) -> int:
    """The question-grid's 1-indexed box number for a template (row, col), in
    normal human reading order: LEFT TO RIGHT within each row, then TOP TO
    BOTTOM. Box 1 is the top-left panel, box 3 is top-right, box 4 is the next
    row's leftmost, and so on through box 9 (bottom-right). Works for any
    (row, col) in the 3x3 template regardless of whether that cell is
    populated — the caller only asks for numbers of panels that exist, so the
    blank '?' cell's number is simply never emitted, not skipped/renumbered
    around."""
    return row * N_GRID + col + 1


def rows_of(panels: list[tuple]) -> list[list[tuple]]:
    """Arrange (row, col, bbox) panels into a row-preserving, ragged nested
    list — row index and column order both come directly from the template
    grid coordinates assigned in panels_in_group(), so no y-clustering or
    x-sorting heuristics are needed here."""
    if not panels:
        return []
    n_rows = max(r for r, _, _ in panels) + 1
    rows: list[list[tuple]] = [[] for _ in range(n_rows)]
    for r, c, b in sorted(panels, key=lambda t: (t[0], t[1])):
        rows[r].append(b)
    return [row for row in rows if row]


# ----------------------------------------------------------------------------- 
# Symbol classification
# ----------------------------------------------------------------------------- 
# The suit is decided from the glyph's VERTICAL WIDTH PROFILE — how wide the
# black shape is near its top vs. its bottom — rather than from scalar contour
# metrics (solidity/perimeter/Hu-moments). Two earlier approaches were tried
# and rejected first:
#   - a cv2.matchShapes template match against externally-rendered font glyphs:
#     the real symbols are tiny (~16x20px) and pixelated, so their Hu-moment
#     signature didn't reliably match a clean glyph (7/11 on ground truth).
#   - hand-tuned solidity/perimeter thresholds: brittle. A spade whose stem was
#     split off by a grid line (now fixed upstream by INK_CLOSE_KERNEL) read as
#     a "club", and solidity alone barely separates heart from spade.
# The width profile is far more robust because it keys directly off the ONE
# feature that actually distinguishes the four suits at this resolution — where
# the mass sits vertically — and each threshold below lands in a wide, visibly
# empty gap in the measured distribution over all 1913 symbols in this pack:
#   * club   — the only glyph wider than tall (3 lobes). aspect w/h clusters at
#              >=1.05 with nothing between 1.0 and 1.05.
#   * heart  — top-heavy: two lobes up top, a point at the bottom, so the top
#              band is much wider than the bottom band (bottom-minus-top < 0,
#              with an empty gap between -0.2 and -0.1).
#   * spade  — bottom-heavy: a point up top, a wide flared base, so the bottom
#              band is wide (>=~0.33 of max width; diamonds never exceed ~0.30).
#   * diamond— symmetric and pointed at BOTH ends, so its bottom band is narrow.
# If you feed in a different rendering (font/resolution), re-check the
# histograms with --debug before trusting these cutoffs.
def _symbol_features(sym_bin: np.ndarray) -> dict:
    """Shape features of one isolated glyph, from its binary crop (rows x cols,
    nonzero = ink). `top`/`bot` are the mean row-width of the top and bottom
    quarter-bands, normalized to the glyph's widest row."""
    h, w = sym_bin.shape
    row_w = (sym_bin > 0).sum(axis=1).astype(float)
    w_max = row_w.max() if row_w.size and row_w.max() else 1.0
    band = max(2, h // 4)
    return {
        "aspect": w / h if h else 0.0,
        "top": float(row_w[:band].mean() / w_max),
        "bot": float(row_w[-band:].mean() / w_max),
        "area": float((sym_bin > 0).sum()),
    }


def classify_suit(f: dict) -> str:
    if f["aspect"] >= 1.0:          # club: the only suit wider than it is tall
        return "club"
    if f["bot"] - f["top"] < -0.15:  # heart: top-heavy (lobes up, point down)
        return "heart"
    if f["bot"] >= 0.33:            # spade: bottom-heavy (point up, wide base)
        return "spade"
    return "diamond"               # symmetric, pointed at both ends


def _merge_line_peaks(sums: np.ndarray, length: int, frac: float = 0.3) -> list[int]:
    """Positions of the strong yellow lines in a 1D projection profile (a
    column-sum or row-sum). `frac` is the fraction of `length` a sum must
    clear to count as "on a line". Anti-aliasing produces a few adjacent
    hot pixels per real line, so consecutive hits are merged into one
    position (their mean) before returning."""
    idx = [i for i in range(len(sums)) if sums[i] > frac * length]
    lines: list[int] = []
    cur: list[int] = []
    for i in idx:
        if cur and i - cur[-1] <= 2:
            cur.append(i)
        else:
            if cur:
                lines.append(int(np.mean(cur)))
            cur = [i]
    if cur:
        lines.append(int(np.mean(cur)))
    return lines


def _yellow_mask(region: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(region.astype(int))
    return ((b < 180) & (g > 200) & (r > 200)).astype(np.uint8) * 255


def _band_candidate_xs(img: np.ndarray, y0: int, y1: int, x_lo: int, x_hi: int) -> list[int]:
    """Candidate vertical-line x positions: yellow pixels are summed down
    columns, scanned across the FULL GROUP width but restricted to one
    panel's own y-band (row). Scanning the full group width rather than
    just one ~150px-wide panel is what makes this reliable — a real grid
    line is shared by every panel in the row, so it gets much stronger,
    more consistent support in the column-sum profile than it would from a
    single panel alone."""
    band = img[y0:y1, x_lo:x_hi]
    col_sums = _yellow_mask(band).sum(axis=0) / 255
    return [xv + x_lo for xv in _merge_line_peaks(col_sums, y1 - y0)]


def _band_candidate_ys(img: np.ndarray, x0: int, x1: int, y_lo: int, y_hi: int) -> list[int]:
    """Transpose of _band_candidate_xs: candidate horizontal-line y
    positions, scanned across the full group height but restricted to one
    panel's own x-band (column)."""
    band = img[y_lo:y_hi, x0:x1]
    row_sums = _yellow_mask(band).sum(axis=1) / 255
    return [yv + y_lo for yv in _merge_line_peaks(row_sums, x1 - x0)]


def _best_interior_window(candidates: list[int], lo: int, hi: int, k: int = 3) -> list[int] | None:
    """Pick k consecutive candidate line positions that best represent THIS
    panel's own interior grid lines, out of a candidate list that may
    contain extra lines bled in from a touching neighboring panel (rows are
    rendered edge-to-edge — see the module docstring). Preference order:
    (1) a window fully inside [lo, hi] (this panel's own bounding box, with
    a small tolerance) over one that isn't, (2) whichever window's mean is
    closest to the panel's center — a neighbor's bled-in line shows up near
    one of the two edges, not centered."""
    if len(candidates) < k:
        return None
    center = (lo + hi) / 2
    best = None
    for i in range(len(candidates) - k + 1):
        window = candidates[i:i + k]
        span_ok = window[0] >= lo - 8 and window[-1] <= hi + 8
        score = (0 if span_ok else 1, abs(sum(window) / k - center))
        if best is None or score < best[0]:
            best = (score, window)
    return best[1]


def panel_interior_vertices(img: np.ndarray, group_bbox: tuple, bbox: tuple) -> tuple[list, list]:
    """The 3 interior reference-grid vertex positions per axis for one
    panel, in local panel pixel coords (i.e. strictly INSIDE the gray area,
    excluding the panel's own outer edges at 0 and w/h).

    Detected from the actual yellow grid pixels (band-scoped per row/column
    — see _band_candidate_xs/_ys — and disambiguated with
    _best_interior_window), NOT geometric guessing. This combination was
    verified to find exactly 3 clean, evenly-spaced interior lines on all
    400 panels across all 25 questions in this pack (both groups). Two
    earlier approaches were tried and rejected first, for context:
      - per-panel-only line detection (scanning just the single panel's own
        crop, the original detect_grid_lines()): unreliable — 231-327/400
        panels (depending on image variant) came back with the wrong line
        count, mostly from a touching neighbor's line bleeding into the
        crop, or the panel's own edge line going undetected.
      - pure geometric quartering (w/4, w/2, 3w/4, no pixel detection at
        all): always produced exactly 3 positions and never crashed, but
        they're only an approximation — the panel bounding box itself has
        a few px of slop — so a symbol genuinely sitting on a real grid
        vertex routinely measured just outside tolerance and got
        misclassified as "cell" instead of "vertex" (this is what broke
        vertex/cell differentiation last time).
    Falls back to quartering only if band detection can't find 3 candidates
    at all (shouldn't happen in this image family, but keeps the reader
    from crashing on an unexpected input)."""
    gx0, gy0, gx1, gy1 = group_bbox
    x, y, w, h = bbox

    xs_cand = _band_candidate_xs(img, y, y + h, gx0, gx1)
    ys_cand = _band_candidate_ys(img, x, x + w, gy0, gy1)
    xs_sel = _best_interior_window(xs_cand, x, x + w)
    ys_sel = _best_interior_window(ys_cand, y, y + h)

    if xs_sel is None or ys_sel is None:
        return [w / 4, w / 2, 3 * w / 4], [h / 4, h / 2, 3 * h / 4]

    return [v - x for v in xs_sel], [v - y for v in ys_sel]


def classify_placement(cx: float, cy: float, xs: list, ys: list,
                        w: float, h: float) -> dict:
    """Decide whether a symbol center sits ON a reference-grid vertex or
    floating inside a cell interior.

    xs/ys are the 3 interior vertex positions from panel_interior_vertices()
    — i.e. only vertices INSIDE the gray area are ever considered, so a
    'vertex' result is always numbered 1-3 per axis (1-indexed: vertex 1 is
    the first interior line, vertex 2 the middle/center one, vertex 3 the
    last). A symbol sitting on the panel's own outer edge is therefore never
    reported as a vertex — it falls through to the 'cell' case below, which
    still uses the outer edges (0, w/h) as bin boundaries and keeps its
    existing 0-indexed (row, col) numbering over the panel's 4 cells.

    The vertex-vs-cell threshold is set relative to the actual measured
    spacing between the detected interior lines (20% of the smallest gap)
    rather than a fixed pixel count, so it adapts to each panel's own
    detected grid instead of assuming perfectly even spacing.
    """
    if len(xs) < 1 or len(ys) < 1:
        return {"placement": "unknown", "grid_coord": None}
    gaps = [b - a for a, b in zip(xs, xs[1:])] + [b - a for a, b in zip(ys, ys[1:])]
    tol = max(3, 0.20 * min(gaps)) if gaps else 3

    nearest_xi = min(range(len(xs)), key=lambda i: abs(xs[i] - cx))
    nearest_yi = min(range(len(ys)), key=lambda i: abs(ys[i] - cy))
    dist_x = abs(xs[nearest_xi] - cx)
    dist_y = abs(ys[nearest_yi] - cy)

    if dist_x < tol and dist_y < tol:
        return {"placement": "vertex", "grid_coord": [nearest_yi + 1, nearest_xi + 1]}

    # not on a vertex -> report which of the 4 CELLs (bounded by the outer
    # edges plus the 3 interior vertices) it falls in; 0-indexed, unchanged
    import bisect
    xs_bounds = [0] + xs + [w]
    ys_bounds = [0] + ys + [h]
    col = min(bisect.bisect_right(xs_bounds, cx) - 1, len(xs_bounds) - 2)
    row = min(bisect.bisect_right(ys_bounds, cy) - 1, len(ys_bounds) - 2)
    return {"placement": "cell", "grid_coord": [row, col]}


def _all_ink_contours(panel_crop: np.ndarray) -> list:
    """Every separate ink blob in a panel, largest-area-first excluded — no
    ordering assumed here; read_cell() sorts them into reading order.
    Previously this kept only the single largest contour, which silently
    discarded every other symbol in a multi-symbol panel."""
    h, w = panel_crop.shape
    cnts, _ = cv2.findContours(panel_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # A genuine symbol always sits with margin from the panel's own outer
    # edge (0/w/h) — it's only ever clipped by an INTERIOR yellow grid line,
    # which reduces its ink area (see INK_CLOSE_KERNEL note) but never moves
    # its bounding box flush against the crop border. A bleed sliver from a
    # touching neighboring panel (see panels_in_group() on why panels aren't
    # separated by a reliable gap) is the opposite: it's a fragment of a
    # symbol that lives mostly outside this crop, so its own bounding box is
    # pinned flush against the edge the neighbor bled in from.
    # This replaces a fixed area floor (tried first, then rejected): the area
    # histogram has real symbols as small as ~110px^2 when a grid vertex
    # passes straight through them (e.g. question-02 boxes 4-6, each losing a
    # 3rd of their area to the crossing lines), overlapping the range where
    # bleed slivers also fall (e.g. question-02 box1's spurious 86px^2
    # "club"), so no single area cutoff can separate the two cleanly.
    real = []
    for c in cnts:
        if cv2.contourArea(c) <= 50:   # still drop anti-aliasing specks
            continue
        sx, sy, sw, sh = cv2.boundingRect(c)
        if sx <= 1 or sy <= 1 or sx + sw >= w - 1 or sy + sh >= h - 1:
            continue
        real.append(c)
    return real


def read_symbol(sym, crop: np.ndarray, w: int, h: int, xs: list, ys: list) -> dict:
    """Classify one already-isolated ink contour: suit, normalized position,
    and grid-relative placement (vertex vs cell interior). `crop` is the
    panel's binary ink mask (panel-local coords), from which this symbol's own
    bounding box is sliced to measure its vertical width profile."""
    sx, sy, sw, sh = cv2.boundingRect(sym)
    f = _symbol_features(crop[sy:sy + sh, sx:sx + sw])
    suit = classify_suit(f)
    cx, cy = sx + sw / 2, sy + sh / 2                # pixel coords, local to panel
    place = classify_placement(cx, cy, xs, ys, w, h)
    return {
        "suit": suit,
        "position": [round(cx / w, 3), round(cy / h, 3)],   # normalized 0..1
        "placement": place["placement"],       # "vertex" | "cell" | "unknown"
        "grid_coord": place["grid_coord"],      # [row, col] index, meaning depends on placement
        "_features": f,
    }


def read_cell(img: np.ndarray, black_mask: np.ndarray, bbox: tuple, group_bbox: tuple) -> dict:
    """Read ALL symbols in one panel (a panel may hold 0, 1, or many suit
    symbols — see the multi-symbol puzzle variant this was extended for).
    Returns {"symbols": [...], "count": n}, with symbols sorted into a
    consistent reading order (top-to-bottom, then left-to-right) so output
    is deterministic regardless of contour-discovery order."""
    x, y, w, h = bbox
    crop = black_mask[y:y + h, x:x + w]
    xs, ys = panel_interior_vertices(img, group_bbox, bbox)

    syms = _all_ink_contours(crop)
    symbols = [read_symbol(s, crop, w, h, xs, ys) for s in syms]
    symbols.sort(key=lambda s: (s["position"][1], s["position"][0]))  # reading order
    return {"symbols": symbols, "count": len(symbols)}


# ----------------------------------------------------------------------------- 
# Top-level read + pretty printing
# ----------------------------------------------------------------------------- 
def read_puzzle(path: str, debug: bool = False) -> dict:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    black_mask = ink_mask_of(img)
    gray_mask = gray_mask_of(img)

    question_bbox, answer_bbox = find_group_bboxes(gray_mask)
    question_panels = panels_in_group(gray_mask, question_bbox)
    answer_panels = panels_in_group(gray_mask, answer_bbox)

    # Question grid: keyed by box_number() (1-indexed, right-to-left then
    # top-to-bottom — see box_number() docstring), NOT by nested row/col
    # position, so each entry's JSON key directly says which box it is.
    question_by_box = {
        box_number(r, c): read_cell(img, black_mask, b, question_bbox)
        for r, c, b in question_panels
    }

    # Answer grid: unchanged — flat {letter: panel} dict, A-H in reading order.
    answer_flat = [b for row in rows_of(answer_panels) for b in row]
    answer_by_letter = {
        letter: read_cell(img, black_mask, b, answer_bbox)
        for letter, b in zip(string.ascii_uppercase, answer_flat)
    }

    if debug:
        print("--- QUESTION GRID panel raw features ---", file=sys.stderr)
        for box in sorted(question_by_box):
            cell = question_by_box[box]
            suits = [(s["suit"], s.get("_features")) for s in cell["symbols"]]
            print(f"  box{box} n={cell['count']} symbols={suits}", file=sys.stderr)
        print("--- ANSWER GRID panel raw features ---", file=sys.stderr)
        for letter, cell in answer_by_letter.items():
            suits = [(s["suit"], s.get("_features")) for s in cell["symbols"]]
            print(f"  {letter} n={cell['count']} symbols={suits}", file=sys.stderr)

    # strip internal debug features from every symbol before returning/serializing,
    # and collapse grid_coord into a "(row, col)" coordinate string
    def clean(cell):
        def fmt_symbol(s):
            coord = s["grid_coord"]
            position = f"({coord[0]}, {coord[1]})" if coord else "unknown"
            return {
                "suit": s["suit"],
                "placement": s["placement"],
                "position": position,
            }
        return {
            "count": cell["count"],
            "symbols": [fmt_symbol(s) for s in cell["symbols"]],
        }

    return {
        "question_grid": {str(box): clean(question_by_box[box])
                           for box in sorted(question_by_box)},
        "answer_grid": {k: clean(v) for k, v in answer_by_letter.items()},
    }


SUIT_LABEL = {"spade": "spades", "heart": "hearts",
             "diamond": "diamonds", "club": "clubs", "none": "none"}


def _format_symbol(s: dict) -> str:
    label = SUIT_LABEL.get(s["suit"], s["suit"])
    tag = f"{s['placement']}{s['position']}" if s["position"] != "unknown" else "?"
    return f"{label} {tag}"


def pretty_print_question_grid(question_grid: dict) -> str:
    """Render the question grid as an aligned ASCII table, one output row
    per template row (boxes 1-3, then 4-6, then 7-9), preserving the
    right-to-left box numbering and gracefully skipping the blank '?' box.
    Each cell may show multiple symbols, space-separated."""
    cell_w = 60
    lines = []
    for row_start in (1, 4, 7):
        cells = []
        boxes_in_row = [row_start, row_start + 1, row_start + 2]
        if not any(str(b) in question_grid for b in boxes_in_row):
            continue
        for box in boxes_in_row:
            cell = question_grid.get(str(box))
            if cell is None:
                cells.append("(blank '?' box)".ljust(cell_w))
                continue
            label = " ".join(_format_symbol(s) for s in cell["symbols"]) or "(empty)"
            cells.append(f"box{box}: {label}".ljust(cell_w))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def pretty_print_answer_grid(answer_grid: dict) -> str:
    lines = []
    for letter, cell in answer_grid.items():
        label = " ".join(_format_symbol(s) for s in cell["symbols"]) or "(empty)"
        lines.append(f"  {letter} (n={cell['count']}): {label}")
    return "\n".join(lines)


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Question screenshots (white gaps) (cleaned)")


# -----------------------------------------------------------------------------
# Notation legend — explains the output format. Printed once at the start of
# a run and saved into the JSON output (under "legend") so the result file is
# self-describing without needing this source file alongside it.
# -----------------------------------------------------------------------------
NOTATION_LEGEND = {
    "top_level": {
        "question_grid": "The question matrix: a flat {box_number: panel} "
            "dict, keyed by a 1-indexed box number 1-9 in normal reading "
            "order: LEFT TO RIGHT within a row, then TOP TO BOTTOM. Box 1 is "
            "the top-left panel, box 3 is top-right, box 4 is the next row's "
            "leftmost panel, ..., box 9 is bottom-right. Only 8 of the 9 "
            "keys are present — the outer matrix is 3x3 with exactly one "
            "cell blank (the '?' to be solved for), so that box's number is "
            "simply missing from the dict, not remapped.",
        "answer_grid": "The lettered answer choices, as a flat {letter: "
            "panel} dict, A through H, in reading order (row-major, "
            "top-to-bottom then left-to-right — UNCHANGED from question_grid's "
            "numbering, answer_grid keeps its original letter labels).",
    },
    "panel_fields": {
        "count": "Number of separate ink symbols found in this panel (0, 1, or more).",
        "symbols": "List of symbols found in the panel, each described by "
            "suit/placement/position below.",
    },
    "suit": "One of 'spades', 'hearts', 'diamonds', 'clubs' — the card suit "
        "the ink symbol was classified as (see classify_suit()).",
    "placement": {
        "vertex": "The symbol sits ON one of the panel's reference-grid "
            "vertices — an intersection of the yellow grid lines strictly "
            "INSIDE the gray-filled area (never the panel's own outer edge).",
        "cell": "The symbol floats inside one of the panel's 4 grid cells, "
            "not on a vertex.",
        "unknown": "Grid lines could not be determined for this panel.",
    },
    "position": {
        "note": "Formatted as \"(row, col)\" — meaning depends on placement:",
        "vertex": "1-INDEXED, each of row/col in 1-3. Refers to the panel's "
            "3 interior reference-grid vertices along that axis: 1 = first "
            "(nearest the panel's start edge), 2 = middle/center, 3 = last "
            "(nearest the panel's end edge). Only vertices strictly inside "
            "the gray area are ever used, so a vertex position is always in "
            "this 1-3 range, never on the panel's own outer border.",
        "cell": "0-INDEXED, each of row/col in 0-3. Refers to which of the "
            "panel's 4 grid cells (bounded by the outer edges plus the 3 "
            "interior vertices) the symbol's center falls into.",
    },
}


def _format_legend(legend: dict) -> str:
    lines = ["=== Notation legend ==="]
    lines.append("Top-level structure:")
    for k, v in legend["top_level"].items():
        lines.append(f"  {k}: {v}")
    lines.append("Per-panel fields:")
    for k, v in legend["panel_fields"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"suit: {legend['suit']}")
    lines.append("placement:")
    for k, v in legend["placement"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"position: {legend['position']['note']}")
    lines.append(f"  when placement=vertex: {legend['position']['vertex']}")
    lines.append(f"  when placement=cell:   {legend['position']['cell']}")
    return "\n".join(lines)


def _print_result(result: dict) -> None:
    print("=== Question grid ===")
    print(pretty_print_question_grid(result["question_grid"]))
    print("\n=== Answers ===")
    print(pretty_print_answer_grid(result["answer_grid"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read suit-symbol grid IQ puzzles.")
    ap.add_argument("image", nargs="?",
                     help="single image to read; omit to batch-process every file "
                          "in png_files (found under --dir)")
    ap.add_argument("--dir", default=DEFAULT_DIR,
                     help=f"directory containing the question PNGs for batch mode "
                          f"(default: {DEFAULT_DIR})")
    ap.add_argument("--debug", action="store_true", help="print raw per-panel features")
    ap.add_argument("--json", default="all_results.json",
                     help="path to write the JSON result(s) to")
    args = ap.parse_args()

    # Printed first, and saved into the JSON output ("legend" key) so the
    # result file explains its own notation without needing this script.
    print(_format_legend(NOTATION_LEGEND))
    print()

    if args.image:
        result = read_puzzle(args.image, debug=args.debug)
        _print_result(result)
        with open(args.json, "w") as f:
            json.dump({"legend": NOTATION_LEGEND, "result": result}, f, indent=2)
        print(f"\nFull structured result written to {args.json}")
    else:
        all_results: dict[str, dict] = {}
        for filename in png_files:
            path = os.path.join(args.dir, filename)
            print(f"\n########## {filename} ##########")
            try:
                result = read_puzzle(path, debug=args.debug)
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                all_results[filename] = {"error": str(e)}
                continue
            _print_result(result)
            all_results[filename] = result

        with open(args.json, "w") as f:
            json.dump({"legend": NOTATION_LEGEND, "results": all_results}, f, indent=2)
        ok = sum(1 for r in all_results.values() if "error" not in r)
        print(f"\n{ok}/{len(png_files)} images processed successfully. "
              f"All results written to {args.json}")