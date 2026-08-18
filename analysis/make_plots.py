"""
Emit LaTeX/pgfplots figures for Chapter 5, built directly as TikZ source
(pure string templates over analysis/summary_table.csv - no matplotlib/
tikzplotlib conversion involved) so the numbers are typeset in the
document's own fonts. Reads only summary_table.csv; never touches
test_result/final_result/.

  1. combo_tau08 / combo_tau09  - same method (BeamSearch), before vs. after
     refinement: states (bars, left axis) plus training/validation agreement
     (hollow=initial, filled=final markers, right axis), one figure per
     threshold, all six tasks.
  2. comparison                 - same task, method vs. method: BeamSearch
     vs. SA/GA/PSO across all six tasks, one combined 4-panel groupplot
     (final DFA size, training agreement, validation agreement, runtime) -
     mirrors how Table 5.5 presents every task in one table.

Each .tex is a full \\begin{figure}...\\end{figure} block - \\input{} it
directly into the thesis body. Preamble requirements:
    \\usepackage{pgfplots}
    \\pgfplotsset{compat=1.18}
    \\usepgfplotslibrary{groupplots}   (comparison only)

A .png preview of each figure is also rendered (via pdflatex + pdftoppm, if
both are on PATH) for a quick look outside a LaTeX build - not meant for
inclusion in the thesis itself, which should \\input{} the .tex.
"""
from __future__ import annotations

import os
import csv
import math
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "summary_table.csv")
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

METHODS = ["BeamSearch", "SA", "GA", "PSO"]
METHOD_COLOR = {
    "BeamSearch": "#2a78d6",  # blue
    "SA": "#eb6834",          # orange
    "GA": "#1baf7a",          # aqua
    "PSO": "#eda100",         # yellow
}
METHOD_LABEL = {"BeamSearch": "Beam", "SA": "SA", "GA": "GA", "PSO": "PSO"}
METHOD_MARK = {"BeamSearch": "*", "SA": "square*", "GA": "triangle*", "PSO": "diamond*"}
METHOD_MARK_SIZE = {"BeamSearch": "2.2pt", "SA": "2.0pt", "GA": "2.3pt", "PSO": "2.3pt"}
# Manual x-offsets around each integer task slot (numeric x, not symbolic) so
# panels (a)/(b)/(c)/(d) of comparison_figure line up in the same columns.
METHOD_OFFSET = {"BeamSearch": -0.18, "SA": -0.06, "GA": 0.06, "PSO": 0.18}

# Six tasks that appear in test_result/final_result, regular domain first.
TASKS = ["DocumentReleaseWorkflow", "MultiObligationOrder", "SecureHandshake", "ECG", "mnist", "wafer"]
SHORT_LABEL = {
    "DocumentReleaseWorkflow": "Document",
    "MultiObligationOrder": "Obligation",
    "SecureHandshake": "Handshake",
    "ECG": "ECG5000",
    "mnist": "MNIST",
    "wafer": "Wafer",
}

INITIAL_COLOR = "#898781"           # muted grey - "before" bars
FINAL_COLOR = METHOD_COLOR["BeamSearch"]
TRAIN_AGREEMENT_COLOR = "#1baf7a"   # teal - distinct from FINAL_COLOR and VAL below
VAL_AGREEMENT_COLOR = "#eb6834"     # orange


def load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for k in ("threshold", "train_init", "train_final", "val_init", "val_final", "time_s"):
            r[k] = float(r[k])
        for k in ("batch_size", "final_states"):
            r[k] = int(r[k])
        r["initial_states"] = int(r["initial_states"]) if r["initial_states"] else None
        r["clf_test"] = float(r["clf_test"]) if r["clf_test"] else None
    return rows


def _row(rows, task, method, threshold, batch_size):
    for r in rows:
        if (r["automaton"] == task and r["method"] == method
                and r["threshold"] == threshold and r["batch_size"] == batch_size):
            return r
    return None


def _hex_to_rgb(hexcolor: str) -> tuple[int, int, int]:
    h = hexcolor.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _color_defs(named_colors: dict) -> str:
    lines = [f"\\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}" for name, (r, g, b) in
             ((n, _hex_to_rgb(h)) for n, h in named_colors.items())]
    return "\n".join(lines)


def _snap_agreement_range(values, step: float = 0.05, top_pad_steps: int = 1,
                           bottom_pad_steps: int = 0) -> tuple[float, float]:
    """Tightest [ymin, ymax] on the 0.05 grid that contains `values`, plus one
    extra grid step of headroom above the max (capped at 1.02, since
    agreement can't exceed 1.0). `bottom_pad_steps` adds the same below the
    min - useful when this axis is overlaid on another with independent
    data, where a low value would otherwise land right in the pixel band
    the other axis's small values/labels occupy."""
    lo = math.floor(min(values) / step) * step - bottom_pad_steps * step
    hi = math.ceil(max(values) / step) * step + top_pad_steps * step
    return round(max(lo, 0.0), 4), round(min(hi, 1.02), 4)


# ---------------------------------------------------------------------------
# Figures 1-2 (Table 5.4): states before/after (bars) + train/val agreement
# (markers) against the threshold line, one figure per threshold.
# ---------------------------------------------------------------------------
def combo_figure(rows, threshold: float, batch_size: int = 1000) -> str:
    recs = {t: _row(rows, t, "BeamSearch", threshold, batch_size) for t in TASKS}
    missing = [t for t, r in recs.items() if r is None]
    if missing:
        raise ValueError(f"combo_figure(tau={threshold}): no BeamSearch row for {missing}")

    labels = ",".join(SHORT_LABEL[t] for t in TASKS)
    initial_coords = " ".join(f"({SHORT_LABEL[t]},{recs[t]['initial_states']})" for t in TASKS)
    final_coords = " ".join(f"({SHORT_LABEL[t]},{recs[t]['final_states']})" for t in TASKS)
    train_init_coords = " ".join(f"({SHORT_LABEL[t]},{recs[t]['train_init']:.4f})" for t in TASKS)
    val_init_coords = " ".join(f"({SHORT_LABEL[t]},{recs[t]['val_init']:.4f})" for t in TASKS)
    train_coords = " ".join(f"({SHORT_LABEL[t]},{recs[t]['train_final']:.4f})" for t in TASKS)
    val_coords = " ".join(f"({SHORT_LABEL[t]},{recs[t]['val_final']:.4f})" for t in TASKS)
    train_connectors = "\n".join(
        f"\\addplot[trainColor, thin, opacity=0.55, forget plot] coordinates "
        f"{{({SHORT_LABEL[t]},{recs[t]['train_init']:.4f}) ({SHORT_LABEL[t]},{recs[t]['train_final']:.4f})}};"
        for t in TASKS
    )
    val_connectors = "\n".join(
        f"\\addplot[valColor, thin, opacity=0.55, forget plot] coordinates "
        f"{{({SHORT_LABEL[t]},{recs[t]['val_init']:.4f}) ({SHORT_LABEL[t]},{recs[t]['val_final']:.4f})}};"
        for t in TASKS
    )

    max_states = max(recs[t]["initial_states"] for t in TASKS)
    # Generous headroom (bars fill only the bottom ~45% of the panel) so the
    # "before/after" bars and the agreement markers/labels above never land
    # in the same pixel band - the two y-axes are independent scales, and
    # without this a tall bar can coincidentally sit right where an
    # agreement value/label needs to go.
    y_max_states = (int(max_states * 2.8) // 5 + 1) * 5
    all_agreement = [recs[t][k] for t in TASKS for k in ("train_init", "train_final", "val_init", "val_final")]
    y_min_agreement = min(0.6, round(min(all_agreement) - 0.05, 2))

    colors = _color_defs({
        "initColor": INITIAL_COLOR,
        "finalColor": FINAL_COLOR,
        "trainColor": TRAIN_AGREEMENT_COLOR,
        "valColor": VAL_AGREEMENT_COLOR,
    })
    tau_tag = str(threshold).replace(".", "")

    return f"""{colors}
\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
% ===== Left axis: initial and final states =====
% Carries the ONE combined legend for both axes: two overlaid axes each
% positioning their own legend via `at=` fight over the same coordinate
% space and silently clobber each other, so the right axis's series get
% phantom \\addlegendimage entries here instead of a legend of their own.
\\begin{{axis}}[
    width=13cm, height=7cm,
    ybar,
    bar width=9pt,
    axis y line*=left,
    ylabel={{Number of states}},
    ymin=0, ymax={y_max_states},
    symbolic x coords={{{labels}}},
    xtick=data,
    x tick label style={{rotate=20, anchor=east}},
    enlarge x limits=0.12,
    legend style={{at={{(0.5,1.42)}}, anchor=south, legend columns=3, draw=none, font=\\small, column sep=8pt}},
    legend cell align=left,
    nodes near coords,
    every node near coord/.append style={{font=\\footnotesize}},
]
\\addplot[fill=initColor!35!white, draw=initColor, line width=0.5pt] coordinates {{{initial_coords}}};
\\addlegendentry{{Initial (RPNI)}}
\\addplot[fill=finalColor!70!white, draw=finalColor, line width=0.5pt] coordinates {{{final_coords}}};
\\addlegendentry{{Final (refined)}}
\\addlegendimage{{gray!65, dashed, line width=0.8pt}}
\\addlegendentry{{Threshold ($\\tau={threshold}$)}}
\\addlegendimage{{only marks, color=trainColor, mark=diamond, mark size=2.8pt}}
\\addlegendentry{{Training agreement (initial)}}
\\addlegendimage{{only marks, color=trainColor, mark=diamond*, mark size=2.8pt}}
\\addlegendentry{{Training agreement (final)}}
\\addlegendimage{{only marks, color=valColor, mark=square, mark size=2.4pt}}
\\addlegendentry{{Validation agreement (initial)}}
\\addlegendimage{{only marks, color=valColor, mark=square*, mark size=2.4pt}}
\\addlegendentry{{Validation agreement (final)}}
\\end{{axis}}

% ===== Right axis: training / validation agreement, before vs. after =====
% Hollow marker = initial (origin RPNI automaton), filled marker = final
% (after refinement); the thin connecting segment makes the before/after
% change at each task read the same way the states bars do on the left axis.
\\begin{{axis}}[
    width=13cm, height=7cm,
    axis y line*=right,
    axis x line=none,
    ylabel={{Agreement}},
    ymin={y_min_agreement}, ymax=1.02,
    symbolic x coords={{{labels}}},
    xtick=data,
    enlarge x limits=0.12,
]
\\addplot[color=gray!65, dashed, line width=0.8pt, forget plot]
    coordinates {{({SHORT_LABEL[TASKS[0]]},{threshold}) ({SHORT_LABEL[TASKS[-1]]},{threshold})}};
{train_connectors}
{val_connectors}
\\addplot[only marks, color=trainColor, mark=diamond, mark size=2.8pt, forget plot]
    coordinates {{{train_init_coords}}};
\\addplot[only marks, color=valColor, mark=square, mark size=2.4pt, forget plot]
    coordinates {{{val_init_coords}}};
\\addplot[only marks, color=trainColor, mark=diamond*, mark size=2.8pt, forget plot,
    nodes near coords={{\\pgfmathprintnumber[fixed, fixed zerofill, precision=4]{{\\pgfplotspointmeta}}}},
    every node near coord/.append style={{font=\\footnotesize, text=trainColor, anchor=south east, xshift=-4pt, yshift=3pt}}]
    coordinates {{{train_coords}}};
\\addplot[only marks, color=valColor, mark=square*, mark size=2.4pt, forget plot,
    nodes near coords={{\\pgfmathprintnumber[fixed, fixed zerofill, precision=4]{{\\pgfplotspointmeta}}}},
    every node near coord/.append style={{font=\\footnotesize, text=valColor, anchor=north west, xshift=4pt, yshift=-3pt}}]
    coordinates {{{val_coords}}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{DFA size and agreement before and after Beam Search refinement across six
tasks (τ = {threshold}).}}
\\label{{fig:combo-tau{tau_tag}}}
\\end{{figure}}
"""


# ---------------------------------------------------------------------------
# Figure 3 (Table 5.5): one combined figure, all six tasks, BeamSearch vs.
# SA/GA/PSO - a 4-panel groupplot (final states, training agreement,
# validation agreement, runtime), mirroring how Table 5.5 itself presents
# every task in one table. States/agreement/runtime don't share a y-axis
# (log-scale seconds, a 0-1 ratio, and a small integer count aren't
# comparable scales), so each gets its own panel rather than being crammed
# onto shared axes.
# ---------------------------------------------------------------------------
def comparison_figure(rows, threshold: float = 0.8, batch_size: int = 1000) -> str:
    by_method = {t: {m: _row(rows, t, m, threshold, batch_size) for m in METHODS} for t in TASKS}
    missing = [(t, m) for t in TASKS for m in METHODS if by_method[t][m] is None]
    if missing:
        raise ValueError(f"comparison_figure(tau={threshold}): missing rows for {missing}")

    def series(key, fmt):
        out = {}
        for m in METHODS:
            pts = [f"({i + METHOD_OFFSET[m]:.2f},{fmt(by_method[TASKS[i - 1]][m][key])})"
                   for i in range(1, len(TASKS) + 1)]
            out[m] = " ".join(pts)
        return out

    states = series("final_states", lambda v: str(int(v)))
    train_agreement = series("train_final", lambda v: f"{v:.4f}")
    val_agreement = series("val_final", lambda v: f"{v:.4f}")
    runtime = series("time_s", lambda v: f"{v:.1f}")

    all_states = [by_method[t][m]["final_states"] for t in TASKS for m in METHODS]
    all_train = [by_method[t][m]["train_final"] for t in TASKS for m in METHODS]
    all_val = [by_method[t][m]["val_final"] for t in TASKS for m in METHODS]
    all_runtime = [by_method[t][m]["time_s"] for t in TASKS for m in METHODS]
    y_max_states = (max(all_states) // 5 + 2) * 5
    # Each agreement panel gets its own tight range, snapped to the 0.05 grid
    # with one extra grid step of headroom at the top, instead of sharing one
    # range sized for the wider of the two - training agreement clusters much
    # tighter than validation agreement, and a shared range leaves it flat.
    y_min_train, y_max_train = _snap_agreement_range(all_train)
    y_min_val, y_max_val = _snap_agreement_range(all_val)
    y_min_runtime = max(1, round(min(all_runtime) * 0.7))
    y_max_runtime = round(max(all_runtime) * 1.3)

    xtick_labels = ",".join(SHORT_LABEL[t] for t in TASKS)
    colors = _color_defs({f"{m.lower()}Color": METHOD_COLOR[m] for m in METHODS})

    bar_plots = "\n".join(
        f"\\addplot[fill={m.lower()}Color, draw={m.lower()}Color] coordinates {{{states[m]}}};\n"
        f"\\addlegendentry{{{METHOD_LABEL[m]}}}"
        for m in METHODS
    )

    def agreement_panel_plots(coords_by_method):
        return "\n".join(
            f"\\addplot[only marks, mark={METHOD_MARK[m]}, mark size={METHOD_MARK_SIZE[m]}, "
            f"{m.lower()}Color] coordinates {{{coords_by_method[m]}}};"
            for m in METHODS
        )

    train_plots = agreement_panel_plots(train_agreement)
    val_plots = agreement_panel_plots(val_agreement)
    runtime_plots = agreement_panel_plots(runtime)
    tau_node = (
        f"\\node[font=\\tiny, text=gray!70, anchor=south east]\n"
        f"    at (axis cs:{len(TASKS) + 0.5 - 0.08},{threshold}) {{$\\tau={threshold}$}};"
    )

    return f"""{colors}
\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
    group style={{group size=1 by 4, vertical sep=1.0cm}},
    width=0.94\\textwidth,
    height=5.2cm,
    xmin=0.5,
    xmax={len(TASKS) + 0.5},
    xtick={{{",".join(str(i) for i in range(1, len(TASKS) + 1))}}},
    xticklabels={{{xtick_labels}}},
    enlarge x limits=false,
    ymajorgrids=true,
    grid style={{gray!20}},
    axis line style={{gray!65}},
    tick style={{gray!65}},
    title style={{at={{(0,1)}}, anchor=south west, font=\\bfseries}},
]

\\nextgroupplot[
    title={{(a) Final DFA size}},
    ylabel={{Final states}},
    ymin=0,
    ymax={y_max_states},
    xticklabels=\\empty,
    ybar,
    bar width=5.5pt,
    legend style={{
        at={{(0.5,1.35)}}, anchor=south, legend columns=4,
        draw=none, font=\\small, column sep=8pt,
    }},
]
{bar_plots}

\\nextgroupplot[
    title={{(b) Training agreement}},
    ylabel={{Agreement}},
    ymin={y_min_train},
    ymax={y_max_train},
    ytick distance=0.05,
    yticklabel style={{font=\\scriptsize}},
    xticklabels=\\empty,
]
\\addplot[gray!65, dashed, line width=0.8pt, forget plot]
    coordinates {{(0.5,{threshold}) ({len(TASKS) + 0.5},{threshold})}};
{train_plots}
{tau_node}

\\nextgroupplot[
    title={{(c) Validation agreement}},
    ylabel={{Agreement}},
    ymin={y_min_val},
    ymax={y_max_val},
    ytick distance=0.05,
    yticklabel style={{font=\\scriptsize}},
    xticklabels=\\empty,
]
\\addplot[gray!65, dashed, line width=0.8pt, forget plot]
    coordinates {{(0.5,{threshold}) ({len(TASKS) + 0.5},{threshold})}};
{val_plots}
{tau_node}

\\nextgroupplot[
    title={{(d) Runtime}},
    ylabel={{Runtime (s)}},
    ymode=log,
    ymin={y_min_runtime},
    ymax={y_max_runtime},
    x tick label style={{rotate=15, anchor=east}},
]
{runtime_plots}
\\end{{groupplot}}
\\end{{tikzpicture}}
\\caption{{Comparison between the proposed method (Beam) and baseline search algorithms (SA, GA, and PSO) across six tasks under batch size 1000 and $\\tau={threshold}$: (a) final DFA size, (b) training agreement, (c) validation agreement, and (d) runtime.}}
\\label{{fig:comparison}}
\\end{{figure}}
"""


_PNG_WRAPPER = r"""\documentclass{article}
\usepackage[margin=1cm,paperwidth=17cm,paperheight=32cm]{geometry}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\pagestyle{empty}
\begin{document}
\input{%s}
\end{document}
"""


def render_png(tex_path: str, png_path: str, dpi: int = 200) -> bool:
    """Compile a figure .tex (pdflatex) and rasterize it (pdftoppm) into a
    quick-look PNG next to the .tex. Best-effort: returns False (and prints
    a warning instead of raising) if the LaTeX toolchain isn't on PATH, or
    if compilation fails - the .tex itself is the real deliverable."""
    if shutil.which("pdflatex") is None or shutil.which("pdftoppm") is None:
        print(f"  [PNG] Skipped {os.path.basename(png_path)}: pdflatex/pdftoppm not found on PATH.")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        body_name = os.path.basename(tex_path)
        shutil.copy(tex_path, os.path.join(tmp, body_name))
        wrapper_path = os.path.join(tmp, "wrapper.tex")
        with open(wrapper_path, "w", encoding="utf-8") as f:
            f.write(_PNG_WRAPPER % body_name)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "wrapper.tex"],
            cwd=tmp, capture_output=True, text=True,
        )
        pdf_path = os.path.join(tmp, "wrapper.pdf")
        if result.returncode != 0 or not os.path.exists(pdf_path):
            print(f"  [PNG] pdflatex failed for {body_name}:\n" + result.stdout[-1500:])
            return False

        png_stem = png_path[:-4] if png_path.endswith(".png") else png_path
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", str(dpi), pdf_path, png_stem],
            cwd=tmp, capture_output=True, text=True, check=True,
        )
    return os.path.exists(png_path)


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} rows from {CSV_PATH}")

    figures = {
        "combo_tau08": combo_figure(rows, threshold=0.8),
        "combo_tau09": combo_figure(rows, threshold=0.9),
        "comparison": comparison_figure(rows, threshold=0.8),
    }
    for name, tex in figures.items():
        tex_path = os.path.join(PLOTS_DIR, f"{name}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex)
        png_path = os.path.join(PLOTS_DIR, f"{name}.png")
        rendered = render_png(tex_path, png_path)
        print(f"{name}: done" + (" (+ .png)" if rendered else ""))

    print(f"\nAll figures written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
