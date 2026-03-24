#!/usr/bin/env python3
"""Interactive syntax score plot: hover/click to see example sentences for that regime.

Builds the same dual-panel (French / Dutch) syntax score plot as syntax_score_{ablation}.png,
but as an HTML file. Hovering over a point shows example (input → prediction) sentences
from the subject-ablated (or chosen ablation) test set for that proportion and language.

Usage:
  python feb_exp/scripts/interactive_plots/interactive_syntax_plot.py \\
    --runs-dir /path/to/runs \\
    --ablation subject \\
    -o output.html
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_EXAMPLES_IN_HOVER = 4
MAX_EXAMPLE_LEN = 50


def load_alignment_df(runs_dir: Path, ablation_type: str) -> pd.DataFrame:
    """Load alignment JSONs from all run dirs into a DataFrame (same as plot_language_alignment)."""
    results = []
    for run_dir in sorted(runs_dir.glob("p*_run*")):
        json_path = run_dir / f"alignment_{ablation_type}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            metrics = json.load(f)
        dir_name = run_dir.name
        parts = dir_name.split("_run")
        prop = float(parts[0][1:]) / 100.0
        run_id = int(parts[1])
        results.append({
            "prop": prop,
            "run_id": run_id,
            "ablation": ablation_type,
            **metrics,
        })
    if not results:
        raise FileNotFoundError(f"No alignment_{ablation_type}.json found in {runs_dir}")
    return pd.DataFrame(results).sort_values("prop")


def build_examples_lookup(
    runs_dir: Path,
    ablation_type: str,
    max_per_regime: int = 10,
):
    """Build (prop, lang) -> list of {input, gold, prediction} from ablation_predictions.csv."""
    from collections import defaultdict
    lookup = defaultdict(list)
    for run_dir in sorted(runs_dir.glob("p*_run*")):
        csv_path = run_dir / "ablation_predictions.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning("Skip %s: %s", csv_path, e)
            continue
        if "ablation" not in df.columns or "language" not in df.columns:
            continue
        sub = df[df["ablation"] == ablation_type]
        if sub.empty:
            continue
        parts = run_dir.name.split("_run")
        prop = float(parts[0][1:]) / 100.0
        for lang in ("fr", "nl"):
            lang_sub = sub[sub["language"] == lang]
            if lang_sub.empty:
                continue
            key = (prop, lang)
            if len(lookup[key]) >= max_per_regime:
                continue
            need = max_per_regime - len(lookup[key])
            for _, row in lang_sub.head(need).iterrows():
                lookup[key].append({
                    "input": str(row.get("input", "")),
                    "gold": str(row.get("gold", "")),
                    "prediction": str(row.get("prediction", "")),
                })
    return dict(lookup)


def format_examples_for_hover(examples: list[dict], max_len: int = MAX_EXAMPLE_LEN) -> str:
    """Turn list of {input, prediction} into HTML-like hover snippet."""
    if not examples:
        return "No example data for this regime."
    lines = []
    for i, ex in enumerate(examples[:MAX_EXAMPLES_IN_HOVER], 1):
        inp = ex.get("input", "")[:max_len]
        pred = ex.get("prediction", "")[:max_len]
        if len(ex.get("input", "")) > max_len:
            inp += "…"
        if len(ex.get("prediction", "")) > max_len:
            pred += "…"
        lines.append(f"{i}. {inp} → {pred}")
    return "<br>".join(lines)


def build_interactive_figure(runs_dir, ablation, metric):
    """Build the interactive Plotly figure (and optionally save to HTML). Returns fig."""
    runs_dir = Path(runs_dir).resolve()
    fr_col = f"fr_{metric}_score"
    nl_col = f"nl_{metric}_score"
    df = load_alignment_df(runs_dir, ablation)
    if fr_col not in df.columns or nl_col not in df.columns:
        raise ValueError("Columns %s / %s not in data (have: %s)" % (fr_col, nl_col, list(df.columns)))
    examples_lookup = build_examples_lookup(runs_dir, ablation)

    # Build hover text per point: one row per run for FR panel, one per run for NL panel
    df = df.sort_values(["prop", "run_id"])
    hover_fr = []
    hover_nl = []
    score_label = {"syntax": "Syntax score", "morphology": "Morphology score", "alignment": "Alignment score"}[metric]
    for _, row in df.iterrows():
        prop, run_id = row["prop"], row["run_id"]
        score_fr = row.get(fr_col)
        score_nl = row.get(nl_col)
        ex_fr = format_examples_for_hover(examples_lookup.get((prop, "fr"), []))
        ex_nl = format_examples_for_hover(examples_lookup.get((prop, "nl"), []))
        hover_fr.append(
            f"<b>Proportion</b> {prop:.2f} | <b>Run</b> {run_id}<br>"
            f"<b>{score_label}</b> {score_fr:.4f}<br><br><b>Examples (FR test):</b><br>{ex_fr}"
        )
        hover_nl.append(
            f"<b>Proportion</b> {prop:.2f} | <b>Run</b> {run_id}<br>"
            f"<b>{score_label}</b> {score_nl:.4f}<br><br><b>Examples (NL test):</b><br>{ex_nl}"
        )

    n_runs = df.groupby("prop").size().iloc[0] if len(df) else 0
    title_suffix = {"syntax": "word order", "morphology": "token share", "alignment": "syntax + morphology"}[metric]
    title = f"Language alignment – {metric} ({title_suffix}) – {ablation} ablation ({n_runs} runs per proportion)"

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("French test sentences", "Dutch test sentences"),
        horizontal_spacing=0.12,
    )

    # Left: FR
    run_ids = df["run_id"].unique()
    for run_id in run_ids:
        run_df = df[df["run_id"] == run_id].sort_values("prop")
        mask = df["run_id"] == run_id
        fig.add_trace(
            go.Scatter(
                x=run_df["prop"],
                y=run_df[fr_col],
                mode="lines",
                line=dict(color="gray", width=1, dash="solid"),
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    # FR scatter (one point per run) with hover
    fig.add_trace(
        go.Scatter(
            x=df["prop"],
            y=df[fr_col],
            mode="markers",
            marker=dict(size=8, color="rgba(31,119,180,0.7)", line=dict(width=0.5, color="white")),
            text=hover_fr,
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    medians_fr = df.groupby("prop")[fr_col].median()
    fig.add_trace(
        go.Scatter(
            x=medians_fr.index,
            y=medians_fr.values,
            mode="lines+markers",
            line=dict(color="black", width=2.5, dash="dash"),
            marker=dict(size=10, color="black", line=dict(width=0.5, color="white")),
            name="Median",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Right: NL
    for run_id in run_ids:
        run_df = df[df["run_id"] == run_id].sort_values("prop")
        fig.add_trace(
            go.Scatter(
                x=run_df["prop"],
                y=run_df[nl_col],
                mode="lines",
                line=dict(color="gray", width=1, dash="solid"),
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=df["prop"],
            y=df[nl_col],
            mode="markers",
            marker=dict(size=8, color="rgba(255,127,14,0.7)", line=dict(width=0.5, color="white")),
            text=hover_nl,
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    medians_nl = df.groupby("prop")[nl_col].median()
    fig.add_trace(
        go.Scatter(
            x=medians_nl.index,
            y=medians_nl.values,
            mode="lines+markers",
            line=dict(color="black", width=2.5, dash="dash"),
            marker=dict(size=10, color="black", line=dict(width=0.5, color="white")),
            name="Median",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Proportion of French in Training Data", row=1, col=1)
    fig.update_xaxes(title_text="Proportion of French in Training Data", row=1, col=2)
    ylabel = {"syntax": "Syntax score (0=NL order, 1=FR order)", "morphology": "Morphology score (0=Dutch, 1=French)", "alignment": "Alignment score (0=Dutch, 1=French)"}[metric]
    fig.update_yaxes(title_text=ylabel, row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text=ylabel, row=1, col=2, range=[0, 1])
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_white",
        width=1200,
        height=520,
        margin=dict(t=80, b=60, l=60, r=40),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Interactive syntax score plot with example sentences on hover.")
    parser.add_argument(
        "--runs-dir",
        "-r",
        type=Path,
        required=True,
        help="Directory containing p*_run* with alignment_*.json and ablation_predictions.csv",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="subject",
        choices=("none", "subject", "verb", "object"),
        help="Ablation type (default: subject)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="syntax",
        choices=("syntax", "morphology", "alignment"),
        help="Metric to plot (default: syntax). alignment = syntax + morphology combined.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: {metric}_score_{ablation}_interactive.html in cwd)",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also (or only) export static PNG for e.g. Google Slides (requires: pip install kaleido)",
    )
    args = parser.parse_args()
    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        parser.error("runs_dir does not exist: %s" % runs_dir)
    try:
        fig = build_interactive_figure(runs_dir, args.ablation, args.metric)
    except FileNotFoundError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e
    except ValueError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e
    out = args.output or Path("%s_score_%s_interactive.html" % (args.metric, args.ablation))
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    logger.info("Wrote %s", out)

    if args.png is not None:
        png_path = Path(args.png).resolve()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(png_path), scale=2)
            logger.info("Wrote PNG %s (use in Google Slides: Insert > Image)", png_path)
        except Exception as e:
            logger.error("PNG export failed (install kaleido: pip install kaleido): %s", e)


if __name__ == "__main__":
    main()
