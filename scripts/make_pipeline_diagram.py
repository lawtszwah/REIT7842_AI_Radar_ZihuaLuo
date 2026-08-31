"""Generate docs/design/pipeline.svg -- the design artefact figure.

Kept as a script rather than a hand-drawn file so the diagram and the code stay
in step: if the protocol changes, the diagram is regenerated from this source.
"""

from __future__ import annotations

from pathlib import Path

W, H = 1520, 900
ACCENT = {"a": "#1d4ed8", "b": "#6d28d9", "c": "#0f766e", "d": "#b45309"}
INK, MUTED, LINE = "#0f172a", "#475569", "#cbd5e1"
FILL, WARN_FILL, WARN_LINE = "#f8fafc", "#fef3c7", "#d97706"
KEY_FILL, KEY_LINE = "#ecfdf5", "#0f766e"

out: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(x, y, w, h, fill=FILL, stroke=LINE, rx=7, dash=None, sw=1.2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
               f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=10, fill=INK, weight="normal", anchor="start", family=None, style=None):
    fam = family or "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
    st = f' font-style="{style}"' if style else ""
    out.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
               f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{st}>{esc(s)}</text>')


def box(x, y, w, h, title, lines, accent, *, fill=FILL, stroke=LINE, dash=None, tag=None):
    rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash)
    out.append(f'<rect x="{x}" y="{y}" width="4.5" height="{h}" rx="2" fill="{accent}"/>')
    text(x + 14, y + 19, title, size=11.5, weight="700")
    ty = y + 36
    for ln in lines:
        mono = ln.startswith("`")
        text(x + 14, ty, ln.strip("`"), size=9.6, fill=MUTED,
             family="'SF Mono', ui-monospace, Menlo, monospace" if mono else None)
        ty += 14.5
    if tag:
        tw = 7.0 * len(tag) + 12
        rect(x + w - tw - 10, y + 9, tw, 17, fill=accent, stroke=accent, rx=8)
        text(x + w - tw / 2 - 10, y + 21, tag, size=9.5, fill="#ffffff", weight="700", anchor="middle")


def arrow(x1, y1, x2, y2, color="#94a3b8", label=None, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" stroke-width="1.8" '
               f'fill="none" marker-end="url(#arw)"{d}/>')
    if label:
        text((x1 + x2) / 2, (y1 + y2) / 2 - 6, label, size=9, fill=MUTED, anchor="middle")


out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
out.append('<defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
           'markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>')
rect(0, 0, W, H, fill="#ffffff", stroke="#ffffff", rx=0)

# ---------------------------------------------------------------- header
text(40, 38, "Benchmarking machine learning detectors against CFAR baselines in passive radar",
     size=21, weight="700")
text(40, 60, "Design artefact — apparatus and evaluation workflow  ·  Zihua Luo (49528923)  ·  "
             "REIT7842  ·  supervisor Dr Lu Zhang", size=11, fill=MUTED)
for i, (label, colour) in enumerate([
        ("RQ1  Pd at fixed Pfa: tuned CFAR vs ML families", ACCENT["c"]),
        ("RQ2  where the gains occur", ACCENT["b"]),
        ("RQ3  accuracy retained under a cost budget", ACCENT["d"])]):
    x = 40 + i * 372
    rect(x, 72, 356, 22, fill="#ffffff", stroke=colour, rx=11)
    text(x + 12, 87, label, size=9.8, fill=colour, weight="600")

COLS = [(40, 300, "1 · Data generation", "a"),
        (376, 300, "2 · Detectors under test", "b"),
        (712, 320, "3 · Shared evaluation protocol", "c"),
        (1068, 412, "4 · Outputs, provenance, governance", "d")]
for x, w, title, k in COLS:
    text(x, 122, title, size=13.5, weight="700", fill=ACCENT[k])
    out.append(f'<path d="M {x} 129 L {x + w} 129" stroke="{ACCENT[k]}" stroke-width="2"/>')

# ---------------------------------------------------------------- column 1
A = ACCENT["a"]
box(40, 145, 300, 104, "SimConfig — swept parameters", [
    "`target range bin`, `Doppler bin`",
    "`SNR (dB)`, `noise power`",
    "`clutter patches`, `n targets`, `seed`",
    "Grid and ranges fixed per config file",
], A)
box(40, 265, 300, 132, "Simulator, behind an interface", [
    "GroupSimulator — research group's",
    "parameterised RD-map function",
    "⚠ NOT YET RECEIVED  (RSK-01)",
    "",
    "StubSimulator — stand-in, lets the",
    "pipeline run today; not reportable",
], A, fill=WARN_FILL, stroke=WARN_LINE, tag="ADR-0002")
box(40, 413, 300, 88, "Dataset", [
    "(N, R, D) linear-power maps",
    "per-cell target mask",
    "exact ground truth (range, Doppler, SNR)",
], A)
box(40, 517, 300, 118, "Splits — disjoint seeds", [
    "Train  `seed`      ·  Val  `5000+seed`",
    "Test in-distribution  `1000+seed`",
    "Test shifted regimes: SNR, noise and",
    "clutter outside the training grid",
    "Datasets regenerated, never stored",
], A, tag="RQ2")

# ---------------------------------------------------------------- column 2
B = ACCENT["b"]
box(376, 145, 300, 86, "Equal-budget tuning", [
    "Identical trial count for every arm,",
    "CFAR included — scored on Val at",
    "Pd @ Pfa = 1e-3. Guards against a",
    "straw-man baseline  (RSK-03)",
], B)
box(376, 247, 300, 78, "CFAR family — baselines", [
    "CA-CFAR · OS-CFAR · GO-CFAR",
    "score = cell / local noise estimate,",
    "monotone in the threshold multiplier",
], B)
box(376, 341, 300, 66, "Feature family", [
    "Multi-scale local statistics (3 window",
    "scales, row/column, Doppler geometry)",
    "→ gradient-boosted trees",
], B)
box(376, 423, 300, 74, "Deep family", [
    "Dilated fully-convolutional net (per-cell",
    "logit, sized for real-time inference)",
    "Patch-attention arm — milestone M3",
], B)
box(376, 513, 300, 122, "One interface, no thresholds", [
    "`Detector.score(maps) → (N, R, D)`",
    "Every detector returns a continuous",
    "per-cell statistic. The harness owns the",
    "threshold, so CFAR and ML are compared",
    "at the same operating point, not at",
    "whichever default each ships with.",
], B, fill=KEY_FILL, stroke=KEY_LINE, tag="ADR-0001")

# ---------------------------------------------------------------- column 3
C = ACCENT["c"]
box(712, 145, 320, 118, "Score partition", [
    "Per-target peak within ±1 bin (straddle loss)",
    "Background cells only, ±3-bin exclusion",
    "around every true target",
    "±2 Doppler bins guarded at zero Doppler —",
    "direct-path leakage must not set the threshold",
], C, tag="ADR-0003")
box(712, 279, 320, 66, "Threshold sweep", [
    "Threshold swept over empirical background",
    "quantiles → exact (Pfa, Pd) curve per arm",
], C)
box(712, 361, 320, 152, "Metrics — radar terms, not accuracy", [
    "Pd per target at fixed per-cell Pfa  → RQ1",
    "ROC across Pfa = 1e-5 … 1e-2",
    "Pd stratified by SNR, range, Doppler,",
    "clutter density  → RQ2",
    "Parameter count and ms per map, CPU  → RQ3",
    "5 seeds → mean ± sd on every number",
], C)
box(712, 529, 320, 106, "Single code path", [
    "`evaluation/protocol.py` runs every arm",
    "identically; arms differ only in `score()`.",
    "Protocol frozen at M2 — before the deep",
    "models are tuned. Later changes require an",
    "ADR and a re-run of every arm  (RSK-04)",
], C, fill=KEY_FILL, stroke=KEY_LINE)

# ---------------------------------------------------------------- column 4
D = ACCENT["d"]
box(1068, 145, 412, 118, "results.json — provenance with every number", [
    "Full spec snapshot (data, detector, protocol configs)",
    "git commit · Python / NumPy versions · platform",
    "Simulator name and version — stub runs flagged unusable",
    "Per-seed results retained, not only the aggregate",
], D)
box(1068, 279, 412, 92, "Figures — one script, one results file", [
    "ROC per detector family",
    "Pd against target SNR (where detections are won or lost)",
    "Pd against inference latency (the cost-constrained frontier)",
], D)
box(1068, 387, 412, 84, "experiments/<id>/", [
    "Config snapshot · results.json · figures · run notes",
    "One GitHub issue per experiment, one PR per change;",
    "protocol changes need the `needs-adr` template",
], D)
box(1068, 487, 412, 148, "Reproducibility contract", [
    "A result is the triple (config, seed, commit).",
    "Datasets are a deterministic function of that triple, so",
    "storing the config is storing the data — no data repository",
    "needed, and a figure can be regenerated years later.",
    "",
    "CI runs the tests and the full smoke benchmark on every",
    "push, so a change that breaks the protocol fails first.",
], D, fill=KEY_FILL, stroke="#65a30d")

# ---------------------------------------------------------------- flow arrows
arrow(340, 305, 372, 190, label=None)
arrow(340, 455, 372, 380)
arrow(676, 390, 708, 300)
arrow(1032, 440, 1064, 320)
arrow(676, 574, 708, 574, dash="5 4")
text(692, 566, "per-cell scores", size=8.8, fill=MUTED, anchor="middle")
text(354, 262, "maps", size=8.8, fill=MUTED, anchor="middle")
text(354, 425, "train / val", size=8.8, fill=MUTED, anchor="middle")
text(692, 352, "tuned arms", size=8.8, fill=MUTED, anchor="middle")
text(1048, 388, "metrics", size=8.8, fill=MUTED, anchor="middle")

# ---------------------------------------------------------------- footer bands
rect(40, 665, 1000, 92, fill="#ffffff", stroke=LINE)
text(56, 686, "Tools, constraints and assumptions", size=11.5, weight="700")
foot = [
    "Python 3.11 · NumPy / SciPy · scikit-learn · PyTorch (CPU default) · Matplotlib · pytest · ruff · GitHub Actions · Zotero + IEEE",
    "Constraints: no measured passive-radar data — findings are conditional on the group's simulator; latency measured on one CPU, reported with the machine;",
    "compute bounded by config (map size, dataset size, seed count), so the study scales to the hardware available rather than being redesigned.",
    "Assumptions: ground truth is exact; a detection within ±1 bin is a detection; clutter cancellation and direct-path removal happen upstream of every arm.",
]
for i, ln in enumerate(foot):
    text(56, 706 + i * 13.5, ln, size=9.3, fill=MUTED)

rect(1068, 665, 412, 92, fill="#ffffff", stroke=LINE)
text(1084, 686, "Ethics, governance, security", size=11.5, weight="700")
for i, ln in enumerate([
        "Synthetic data only — no human participants, no personal data,",
        "no transmission. Not human-subject research; confirmed with supervisor.",
        "Group's simulator is never committed or pushed — licence and embargo",
        "govern release; the benchmark can be published even if the simulator cannot.",
]):
    text(1084, 706 + i * 13.5, ln, size=9.3, fill=MUTED)

# ---------------------------------------------------------------- milestones
rect(40, 775, 1440, 86, fill="#f8fafc", stroke=LINE)
text(56, 796, "Plan of work", size=11.5, weight="700")
ms = [("M1", "Protocol, metrics, CFAR baselines,\nstub simulator, CI", "done"),
      ("M2", "Group simulator integrated via\nadapter; protocol frozen", "in progress"),
      ("M3", "Learned arms tuned at equal budget;\nattention arm added", "scheduled"),
      ("M4", "Shift tests; cost-constrained\nsweep", "scheduled"),
      ("M5", "Final 5-seed runs, figures,\nwrite-up", "scheduled")]
mx = 56
for code, desc, status in ms:
    colour = "#15803d" if status == "done" else (ACCENT["d"] if status == "in progress" else MUTED)
    text(mx, 818, code, size=11, weight="700", fill=colour)
    text(mx + 26, 818, f"({status})", size=9, fill=colour)
    for j, ln in enumerate(desc.split("\n")):
        text(mx, 833 + j * 12.5, ln, size=9.3, fill=MUTED)
    if code != "M5":
        out.append(f'<path d="M {mx + 250} 814 L {mx + 275} 814" stroke="{LINE}" '
                   'stroke-width="1.6" marker-end="url(#arw)"/>')
    mx += 288

out.append("</svg>")

path = Path("docs/design/pipeline.svg")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(out))
print(path)
