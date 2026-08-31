"""Generate the three design diagrams under docs/design/.

Kept as a script rather than hand-drawn files so the diagrams and the code stay
in step: if the protocol or the repository layout changes, regenerate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _svg import (ACCENT, Canvas, FILL, KEY_FILL, KEY_LINE, LINE, MUTED,
                  STORE_FILL, STORE_LINE, WARN_FILL, WARN_LINE)

HEAD_SUB = ("Zihua Luo (49528923) · REIT7842 · supervisor Dr Lu Zhang · "
            "Benchmarking ML detectors against CFAR baselines in passive radar")


# ---------------------------------------------------------------- diagram 1
def detection_pipeline(out: Path) -> Path:
    c = Canvas()
    A, B, C, D, E = (ACCENT[k] for k in "abcde")
    c.header("Signal-processing and evaluation pipeline",
             "Artefact 1 of 3 — how one range-Doppler map becomes one number in the results table. "
             + HEAD_SUB,
             [("RQ1  Pd at fixed Pfa: tuned CFAR vs ML families", C),
              ("RQ2  where the gains occur", B),
              ("RQ3  accuracy retained under a cost budget", D)])

    cols = [(40, 268, "1 · Acquisition", A), (348, 250, "2 · Representation", A),
            (638, 300, "3 · Detection statistic", B), (978, 250, "4 · Score partition", C),
            (1268, 212, "5 · Metrics", C)]
    for x, w, t, col in cols:
        c.column_head(x, w, t, col)

    # 1 acquisition
    c.box(40, 145, 268, 118, "Simulated range-Doppler maps", [
        "Parameterised generator, group-supplied",
        "Swept: target range bin, Doppler bin,",
        "SNR (dB), noise power, clutter density",
        "Ground truth exact and known",
        "⚠ generator not yet received (RSK-01)",
    ], A, fill=WARN_FILL, stroke=WARN_LINE)
    c.box(40, 279, 268, 96, "Upstream, assumed done", [
        "Direct-path removal and clutter",
        "cancellation happen before any",
        "detector runs — the ±2-bin zero-",
        "Doppler guard encodes that assumption",
    ], A, dash="5 4")
    c.box(40, 391, 268, 92, "Dataset", [
        "`(N, R, D) float32` linear power",
        "per-cell target mask",
        "`Target(range_bin, doppler_bin, snr_db)`",
    ], A)

    # 2 representation
    c.box(348, 145, 250, 104, "CFAR family input", [
        "Linear power, no preprocessing.",
        "Any transform here would change",
        "the baseline into something other",
        "than the published detector.",
    ], A)
    c.box(348, 265, 250, 104, "Feature family input", [
        "`log10(power)`, then multi-scale",
        "local statistics at three window",
        "sizes, plus row/column means and",
        "distance from zero Doppler",
    ], A)
    c.box(348, 385, 250, 98, "Deep family input", [
        "`log10(power)`, standardised per map",
        "(RD dynamic range spans decades;",
        "without this the networks do not",
        "train)",
    ], A)

    # 3 detectors
    c.box(638, 145, 300, 96, "CFAR — tuned baselines", [
        "CA / OS / GO, guard + training window",
        "`score = cell / local noise estimate`",
        "monotone in the threshold multiplier α",
    ], B)
    c.box(638, 257, 300, 82, "Feature-based classifier", [
        "Multi-scale statistics → gradient-",
        "boosted trees, per-cell decision value",
    ], B)
    c.box(638, 355, 300, 82, "Deep detectors", [
        "Dilated fully-convolutional net (logit)",
        "Patch attention — milestone M3",
    ], B)
    c.box(638, 453, 300, 108, "One interface, no thresholds", [
        "`Detector.score(maps) → (N, R, D)`",
        "Every arm returns a continuous per-cell",
        "statistic. The harness owns the",
        "threshold — that is what makes CFAR and",
        "a network comparable at one operating point.",
    ], B, fill=KEY_FILL, stroke=KEY_LINE, tag="ADR-0001")

    # 4 partition
    c.box(978, 145, 250, 132, "Split the score map", [
        "Per-target peak within ±1 bin",
        "(tolerates straddle loss)",
        "Background = all other cells, minus a",
        "±3-bin exclusion around each target",
        "Zero-Doppler ±2 columns removed",
        "from both counts",
    ], C, tag="ADR-0003")
    c.box(978, 293, 250, 96, "Sweep the threshold", [
        "Thresholds taken at background-score",
        "quantiles, so the false-alarm grid is",
        "exact rather than searched",
    ], C)
    c.box(978, 405, 250, 92, "Repeat", [
        "5 seeds: training data, test data and",
        "model initialisation all move together",
        "Mean ± sd reported on every number",
    ], C)

    # 5 metrics
    c.box(1268, 145, 212, 96, "Pd at fixed Pfa", [
        "Per target, at per-cell",
        "Pfa = 10⁻⁵ … 10⁻²",
        "Headline: Pfa = 10⁻³",
    ], C, tag="RQ1")
    c.box(1268, 257, 212, 100, "Stratified Pd", [
        "By target SNR, range,",
        "Doppler and clutter",
        "density — locates the",
        "gain instead of averaging it",
    ], B, tag="RQ2")
    c.box(1268, 373, 212, 88, "Inference cost", [
        "Parameter count and",
        "ms per map, measured",
        "identically for every arm",
    ], D, tag="RQ3")
    c.box(1268, 477, 212, 84, "Interpretation", [
        "Which arm, at what",
        "operating point, under",
        "what cost budget",
    ], C)

    for x1, y1, x2, y2 in [(312, 204, 344, 197), (312, 437, 344, 320),
                           (602, 197, 634, 193), (602, 317, 634, 298),
                           (602, 434, 634, 396), (942, 300, 974, 211),
                           (1232, 211, 1264, 193), (1232, 341, 1264, 300)]:
        c.arrow(x1, y1, x2, y2)

    c.rect(40, 600, 1440, 116, fill=FILL, stroke=LINE)
    c.text(56, 622, "What is fixed before any model is tuned", size=12, weight="700")
    for i, ln in enumerate([
            "One code path — `evaluation/protocol.py` runs every arm through the same generation, partition, sweep and cost measurement; arms differ only in `score()`.",
            "Equal tuning budget — CFAR gets the same number of configuration trials as the networks, on a validation split disjoint from training and test. A baseline left at default settings is the flaw this project measures.",
            "Protocol frozen at milestone M2, before the deep models are tuned. Any later change to metric, tolerance, guard, splits or seeds requires an Architecture Decision Record and a re-run of every arm.",
            "Figures and their axes specified in advance, so they cannot be selected after the fact to suit the numbers.",
    ]):
        c.text(56, 644 + i * 17, ln, size=9.4, fill=MUTED, mono=False)

    c.rect(40, 736, 1440, 124, fill="#ffffff", stroke=LINE)
    c.text(56, 758, "Tools, constraints and assumptions", size=12, weight="700")
    for i, ln in enumerate([
            "Tools — Python 3.11 · NumPy / SciPy · scikit-learn · PyTorch (CPU by default) · Matplotlib · pytest · ruff · GitHub Actions · Zotero with IEEE style.",
            "Constraints — no measured passive-radar data, so every finding is conditional on the group's simulator and is reported as such; latency measured on one machine and reported with it;",
            "compute bounded by configuration (map size, dataset size, seed count), so the study scales to available hardware rather than being redesigned.",
            "Assumptions — ground truth is exact; a detection within ±1 bin counts as a detection; direct-path removal and clutter cancellation are performed upstream of every arm, identically.",
            "Out of scope — designing a new detector architecture, and any claim about measured data. The contribution is the comparison, not a new detector.",
    ]):
        c.text(56, 780 + i * 17, ln, size=9.4, fill=MUTED)

    return c.save(out)


# ---------------------------------------------------------------- diagram 2
def data_workflow(out: Path) -> Path:
    c = Canvas()
    A, B, C, D, E = (ACCENT[k] for k in "abcde")
    c.header("Data handling and provenance workflow",
             "Artefact 2 of 3 — what exists, where it is allowed to live, and what makes a "
             "result reproducible. " + HEAD_SUB)

    lanes = [(150, 132, "Version-controlled", ["permanent, public repository"], "#f0f9ff", A),
             (306, 150, "Transient", ["held in memory only,", "never written to disk"], "#fafafa", MUTED),
             (480, 150, "Restricted or local", ["never in the public", "repository"], "#fffbeb", WARN_LINE)]
    for y, h, name, sub, bg, col in lanes:
        c.rect(200, y, 1280, h, fill=bg, stroke=LINE, rx=9)
        c.text(40, y + 26, name, size=11.5, weight="700", fill=col)
        for i, ln in enumerate(sub):
            c.text(40, y + 44 + i * 14, ln, size=9, fill=MUTED)

    c.box(220, 166, 268, 100, "Run specification", [
        "`configs/data/*.yaml` — swept parameters",
        "`configs/protocol/*.yaml` — metric, guards, seeds",
        "`seed` (integer) · git commit",
        "Kilobytes. Committed, reviewed, permanent.",
    ], A, size=9.2)
    c.box(1180, 166, 300, 100, "Recorded outputs", [
        "`results.json` — spec snapshot, git commit,",
        "library versions, platform, simulator name",
        "and version, per-seed results",
        "Figures · experiment notes · ADRs. Permanent.",
    ], A, size=9.2)

    c.box(520, 322, 220, 118, "Generate", [
        "Deterministic from",
        "(config, seed).",
        "Tens of GB at full",
        "scale — regenerated",
        "on demand, never",
        "stored.",
    ], MUTED, size=9.2)
    c.box(770, 322, 220, 118, "Split", [
        "Train `seed`",
        "Validation `5000+seed`",
        "Test `1000+seed`",
        "Shift test: SNR, noise",
        "and clutter regimes",
        "outside the training grid",
    ], MUTED, size=9.2)
    c.box(1020, 322, 130, 118, "Detect", [
        "Score maps,",
        "also transient",
    ], MUTED, size=9.2)

    c.box(220, 498, 268, 116, "Group's simulation function", [
        "Supplied by the research group.",
        "Its licence and any embargo govern",
        "what may be shared, so it is reached",
        "only through `adapter.py` and is never",
        "committed or pushed. ⚠ RSK-01, not yet received",
    ], WARN_LINE, fill=WARN_FILL, stroke=WARN_LINE, size=9.2)
    c.box(1020, 498, 260, 116, "Model checkpoints", [
        "Hundreds of MB. Local disk and",
        "UQ RDM, listed in `.gitignore`.",
        "Retained to end of project + 5 years.",
        "Not needed to reproduce a result —",
        "the triple (config, seed, commit) is.",
    ], WARN_LINE, size=9.2)

    c.elbow(354, 266, 354, 372)
    c.arrow(354, 372, 516, 372, label="config + seed")
    c.elbow(354, 498, 354, 400)
    c.arrow(354, 400, 516, 400, label=None)
    c.text(400, 462, "generator, via adapter.py", size=8.8, fill=MUTED, anchor="middle")
    c.arrow(744, 381, 766, 381)
    c.arrow(994, 381, 1016, 381)
    c.elbow(1150, 381, 1330, 272)
    c.text(1240, 373, "metrics + cost", size=8.8, fill=MUTED, anchor="middle")
    c.elbow(1085, 440, 1085, 494)
    c.text(1150, 470, "trained weights", size=8.8, fill=MUTED, anchor="middle")

    c.rect(40, 646, 700, 214, fill="#ffffff", stroke=LINE)
    c.text(56, 670, "Why nothing needs a data repository", size=12, weight="700")
    for i, ln in enumerate([
            "A dataset here is a deterministic function of (config, seed). Storing the",
            "configuration therefore *is* storing the data: the same three files and one",
            "integer regenerate it byte for byte, on any machine, years later.",
            "",
            "That single choice removes the largest data-management burden of the project —",
            "tens of gigabytes that would otherwise need storage, backup, versioning and a",
            "retention plan — and replaces it with a few kilobytes under version control.",
            "",
            "It also makes tampering visible: a figure cites a commit, and the commit",
            "regenerates the data. A result that cannot be reproduced this way is not a result.",
    ]):
        c.text(56, 692 + i * 16, ln, size=9.4, fill=MUTED)

    c.rect(768, 646, 712, 214, fill=KEY_FILL, stroke=KEY_LINE)
    c.text(784, 670, "Governance rules this workflow enforces", size=12, weight="700")
    for i, ln in enumerate([
            "1.  The group's simulator is never committed or pushed. The repository is public;",
            "     .gitignore covers its expected location and this is checked at every merge.",
            "2.  No result produced by the stand-in generator is reportable. Every results.json",
            "     records which simulator produced it, and stub runs carry a warning string.",
            "3.  Datasets are never committed — only configs, seeds, results and figures.",
            "4.  Every reported number carries its commit, library versions and platform.",
            "5.  Per-seed results are retained, not only the aggregate, so variance is inspectable.",
            "6.  All data is synthetic: no human participants, no personal data, no transmission,",
            "     therefore no HREC application and no safety risk assessment (confirmed with",
            "     the supervisor and minuted).",
    ]):
        c.text(784, 692 + i * 16, ln, size=9.4, fill=MUTED)

    return c.save(out)


# ---------------------------------------------------------------- diagram 3
def repo_workflow(out: Path) -> Path:
    c = Canvas()
    A, B, C, D, E = (ACCENT[k] for k in "abcde")
    c.header("Repository set-up and development workflow",
             "Artefact 3 of 3 — how work enters the repository, and what stops a change from "
             "silently invalidating a result. " + HEAD_SUB)

    c.column_head(40, 420, "Repository structure", A)
    c.column_head(500, 500, "How a change enters the repository", B)
    c.column_head(1050, 430, "Milestone gates", D)

    tree = [
        ("src/prdbench/", "the package — one module per stage of the pipeline", True),
        ("  simulation/", "interface.py · stub.py · adapter.py", False),
        ("  data/", "generate.py — (config, seed) → maps + labels", False),
        ("  detectors/", "base.py · cfar.py · features.py · cnn.py · attention.py", False),
        ("  evaluation/", "metrics.py · cost.py · protocol.py", False),
        ("  tuning/", "search.py — equal-budget search", False),
        ("configs/", "data / detectors / protocol — a run is a config plus a seed", True),
        ("experiments/", "one directory per run: config snapshot, results.json,", True),
        ("", "figures, notes. Committed; this is the record.", False),
        ("docs/", "design diagrams · decisions/ (ADRs) · risk_register.md", True),
        ("", "data_management_plan.md · reproducibility.md", False),
        ("tests/", "simulator contract · protocol invariants", True),
        (".github/", "workflows/ci.yml · issue templates · PR template", True),
        ("scripts/", "run_benchmark.sh · make_figures.py · make_diagrams.py", True),
    ]
    c.rect(40, 145, 420, 258, fill=FILL, stroke=LINE)
    y = 168
    for name, desc, top in tree:
        if name:
            c.text(56, y, name, size=9.2, mono=True, weight="700" if top else "normal",
                   fill=A if top else MUTED)
        c.text(196, y, desc, size=8.6, fill=MUTED)
        y += 17.5

    c.box(40, 421, 420, 128, "What is deliberately absent", [
        "No datasets — regenerated from (config, seed)",
        "No model checkpoints — local disk and UQ RDM only",
        "No copy of the research group\u2019s simulator — its licence",
        "governs release, and it is reached only via adapter.py,",
        "so the benchmark can be published even if it cannot",
        "No stand-in generator output presented as a finding",
    ], WARN_LINE, fill=WARN_FILL, stroke=WARN_LINE, size=9.2)

    c.box(40, 567, 420, 118, "Decision records — every protocol choice has a reason", [
        "ADR-0001  Detectors return a continuous per-cell score,",
        "never a decision; the harness owns the threshold.",
        "ADR-0002  The simulator sits behind an interface with a",
        "stand-in, so the project was built before it arrived.",
        "ADR-0003  Pd per target, Pfa per cell, zero Doppler guarded.",
    ], C, size=9.2)

    steps = [
        ("Issue", ["One issue per experiment or change. The `experiment`",
                   "template states which research question it answers;",
                   "`protocol-change` is labelled `needs-adr` and warns",
                   "that every completed arm must be re-run."], B),
        ("Branch and commits", ["Work never happens on `main`. Commit messages carry",
                                "the reasoning, so the history records how the design",
                                "was reached, not only what changed."], B),
        ("Pull request", ["Checklist: tests pass · benchmark still runs end to end ·",
                          "no dataset, checkpoint or group simulator committed ·",
                          "does this affect results, and is an ADR required?"], B),
        ("Continuous integration", ["`ruff` lint · `pytest` — simulator contract and protocol",
                                    "invariants · the smoke benchmark must run end to end.",
                                    "A change that breaks the protocol fails here, before it",
                                    "can contaminate a result."], C),
        ("Merge into protected main", ["The experiment record is then committed under",
                                       "`experiments/<id>/` with its commit hash."], B),
    ]
    y = 145
    for i, (title, lines, col) in enumerate(steps):
        h = 38 + 14.5 * len(lines)
        ci = title.startswith("Continuous")
        c.box(500, y, 500, h, f"{i + 1}. {title}", lines, col,
              fill=KEY_FILL if ci else FILL, stroke=KEY_LINE if ci else LINE, size=9.2)
        if i < len(steps) - 1:
            c.arrow(750, y + h + 1, 750, y + h + 18)
        y += h + 20

    gates = [
        ("M1", ["Apparatus: protocol, metrics, CFAR baselines,", "tests, CI, repository set-up"], "complete", "#15803d", False),
        ("M2", ["Group simulator integrated through adapter.py", "\u25b2 PROTOCOL FROZEN HERE"], "in progress", D, True),
        ("M3", ["Every arm tuned at equal budget;", "attention arm added"], "scheduled", MUTED, False),
        ("M4", ["Distribution-shift tests;", "cost-constrained sweep"], "scheduled", MUTED, False),
        ("M5", ["Final 5-seed runs, figures, write-up"], "scheduled", MUTED, False),
    ]
    y = 145
    for code, lines, status, col, frozen in gates:
        h = 38 + 14.5 * len(lines)
        c.box(1050, y, 430, h, f"{code} — {status}", lines, col,
              fill=WARN_FILL if frozen else FILL, stroke=WARN_LINE if frozen else LINE, size=9.2)
        if code != "M5":
            c.arrow(1265, y + h + 1, 1265, y + h + 15)
        y += h + 17

    c.box(1050, 556, 430, 130, "Why the freeze is a gate, not a note", [
        "This project\u2019s criticism of the literature is that evaluation",
        "choices — operating point, metric, baseline tuning — get",
        "settled after the results are in, at which point no one can",
        "say how much of a reported gain is the model. Freezing",
        "before any model is tuned, and requiring an ADR plus a",
        "full re-run to change it, is what stops this project doing",
        "the same thing — enforced by the templates, not intent.",
    ], C, fill=KEY_FILL, stroke=KEY_LINE, size=9.2)

    c.rect(40, 706, 1440, 154, fill=FILL, stroke=LINE)
    c.text(56, 728, "Principal risks and the controls built into this workflow", size=12, weight="700")
    risks = [
        ("RSK-01", "Group\u2019s simulation function arrives late or not at all",
         "Everything is written against simulation/interface.py; the stand-in lets the rest proceed; tests/test_simulator_contract.py is the handover test"),
        ("RSK-03", "Under-tuned CFAR baseline — the exact flaw this project measures",
         "tuning/search.py gives CFAR the same trial budget as the networks, over declared search spaces, with tuning results reported alongside"),
        ("RSK-04", "Results are an artefact of the protocol, not the detector",
         "One code path in evaluation/protocol.py; protocol frozen at M2; any change requires an ADR and a re-run of every arm"),
        ("RSK-05", "Deep models overfit the simulator\u2019s noise and clutter model",
         "Test configs include SNR, noise and clutter regimes outside the training grid, reported separately from in-distribution results"),
        ("RSK-08", "A result cannot be reproduced at write-up time",
         "A result is (config, seed, commit); provenance written into every results.json; CI re-runs the benchmark on every push"),
    ]
    y = 752
    for code, risk, ctrl in risks:
        c.text(56, y, code, size=9.2, weight="700", fill=E)
        c.text(120, y, risk, size=9.2, fill=MUTED)
        c.text(600, y, ctrl, size=9.2, fill=MUTED)
        y += 21

    return c.save(out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "docs/design"
    for fn, name in ((detection_pipeline, "pipeline.svg"),
                     (data_workflow, "data_workflow.svg"),
                     (repo_workflow, "repo_workflow.svg")):
        print(fn(root / name))
