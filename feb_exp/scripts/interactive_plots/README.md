# Interactive plots

Interactive Plotly versions of alignment/syntax plots. Hover (or click) on a point to see example sentences from the data that generated that regime.

## View in the IDE (no browser)

If you're SSH'd in and can't open the HTML in a browser, open **`view_interactive_plot.ipynb`** in this folder, set `RUNS_DIR` / `ABLATION` / `METRIC` in the first cell, and run the cell. The plot renders in the notebook output so you can hover and see examples inside the IDE.

## Requirements

- `plotly` (in repo requirements.txt)
- Runs directory containing `p*_run*` subdirs with `ablation_predictions.csv` and `alignment_*.json`

## Usage

```bash
# From repo root; use netscratch runs dir so ablation_predictions.csv are available
python feb_exp/scripts/interactive_plots/interactive_syntax_plot.py \
  --runs-dir /n/netscratch/dam_lab/Lab/drooryck/codeswitching-llms/feb_exp/results/mar4/version1_plurality_mixing/runs \
  --ablation subject \
  -o feb_exp/results/mar4/mar4-v1-plurality-mixing/syntax_score_subject_interactive.html
```

If CSVs are missing in runs_dir, the plot still renders but hovers will show "No example data for this regime."

## Using the plot in Google Slides

Slides cannot embed interactive HTML. Use a **static image** instead:

1. Export a PNG (same size as the interactive plot, good for slides):
   ```bash
   pip install kaleido   # one-time
   python feb_exp/scripts/interactive_plots/interactive_syntax_plot.py --runs-dir /path/to/runs --ablation object --metric alignment --png alignment_object.png
   ```
2. In Google Slides: **Insert → Image → Upload from computer** (or drag the PNG onto the slide).
3. Optional: add a short link text (e.g. "Interactive version") that points to the HTML file if you host it somewhere (GitHub Pages, Drive link, etc.) so viewers can open the hoverable plot in a browser.
