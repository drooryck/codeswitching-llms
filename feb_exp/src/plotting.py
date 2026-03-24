"""
Visualization utilities for analyzing model-generated outputs in bilingual translation experiments.

This module provides plotting functions to analyze various metrics computed on model-generated
test outputs. Each metric provides insight into different aspects of the model's bilingual
capabilities:

1. Exact Match: Tests perfect reproduction of target sequences
2. Token Accuracy: Measures local token prediction quality
4. Word Order: Analyzes syntactic structure learning
5. Token Distribution: Examines language mixing and separation

All metrics are computed on the model's generated outputs during test time, comparing against
ground truth targets. The plots show how these metrics vary with the proportion of French
training data, helping understand the model's bilingual learning dynamics.

Example:
    ```python
    from pathlib import Path
    import pandas as pd
    from src.plotting import BilingualPlotter

    # Results DataFrame should contain metrics computed on model-generated test outputs
    results_df = pd.DataFrame({
        'prop': [0.2, 0.5, 0.8],
        'fr_exact': [0.75, 0.82, 0.85],  # Exact match on French outputs
        'nl_exact': [0.80, 0.78, 0.72],  # Exact match on Dutch outputs
        ...
    })

    # Create plotter and generate plots
    plotter = BilingualPlotter(results_df, Path("plots"))
    # plotter.plot_word_order() ## now deprecated lmao
    plotter.plot_french_token_dist()

    # Or create all plots at once
    plotter.create_all_plots()
    ```
"""
import logging
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# Configure logging
logger = logging.getLogger(__name__)

class BilingualPlotter:
    """Plotting utilities for analyzing bilingual model outputs.

    This class provides methods to create various visualizations of metrics computed
    on model-generated test outputs. Each plot type reveals different aspects of
    the model's bilingual capabilities and how they vary with training data composition.

    Attributes:
        results_df: DataFrame containing metrics computed on model outputs
        output_dir: Directory where plots will be saved
        style_initialized: Whether plot style has been set up
    """

    def __init__(self, results_df: pd.DataFrame, output_dir: Path):
        """Initialize plotter with results and output directory.

        Args:
            results_df: DataFrame containing metrics computed on model outputs
            output_dir: Directory where plots will be saved
        """
        self.results_df = results_df
        self.output_dir = Path(output_dir)
        self.style_initialized = False
        self.ablation_type = results_df['ablation'].iloc[0] if 'ablation' in results_df else 'none'

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_plot_style(self):
        """Configure consistent plot style."""
        if not self.style_initialized:
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 12
            self.style_initialized = True

    def _plot_metric_by_proportion(self,
                                fr_metrics: list[str],
                                fr_labels: list[str],
                                nl_metrics: list[str],
                                nl_labels: list[str],
                                title: str,
                                ylabel: str,
                                output_path: Path,
                                ylim: Optional[tuple[float, float]] = None) -> None:
        """Create side-by-side line plots showing metric values by French proportion."""
        self.setup_plot_style()

        # Use manual layout (not constrained) to avoid spacing warnings
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        colors = ['#1f77b4', '#ff7f0e', '#808080', '#2ca02c']
        plt.rcParams.update({
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 13,
            "legend.title_fontsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10
        })

        import textwrap
        def wrap_label(label, width=32):
            return "\n".join(textwrap.wrap(label, width))

        def plot_metrics(ax, metrics, labels):
            for i, (col, label) in enumerate(zip(metrics, labels)):
                color = colors[i % len(colors)]
                ax.scatter(self.results_df['prop'], self.results_df[col],
                        color=color, alpha=0.25, s=25)
                medians = self.results_df.groupby('prop')[col].median()
                ax.plot(medians.index, medians.values,
                        color=color, linewidth=2.5, label=wrap_label(label))
                ax.scatter(medians.index, medians.values,
                        color=color, s=80, zorder=5,
                        edgecolor='white', linewidth=0.5)

        plot_metrics(ax1, fr_metrics, fr_labels)
        plot_metrics(ax2, nl_metrics, nl_labels)

        ablation_label = getattr(self, "ablation_type", None)
        ablation_prefix = f"{ablation_label}-ablated " if ablation_label and ablation_label != "none" else ""

        # Plot subtitles (wrapped)
        for ax, subtitle, tint in [
            (ax1, f"French {ablation_prefix}test sentences", "#1f77b4"),  # bluish tint
            (ax2, f"Dutch {ablation_prefix}test sentences", "#ff7f0e")   # orangish tint
        ]:
            subtitle_wrapped = subtitle.replace(" test sentences", "\ntest sentences")
            ax.set_title(subtitle_wrapped.capitalize(), fontsize=13, pad=18, fontweight="semibold")
            ax.set_xlabel("Proportion of French in Training Data")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            if ylim:
                ax.set_ylim(ylim)

            # Legends with tinted boxes
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=2,
                frameon=True,
                fancybox=True,
                framealpha=0.25,
                edgecolor=tint,
                facecolor=tint,   # box tint
                handlelength=2.5,
                fontsize=13
            )
            # Soften tint by lowering alpha manually
            legend.get_frame().set_alpha(0.15)
            legend.get_frame().set_linewidth(1.0)

        # Global title and layout
        n_runs = self.results_df.groupby("prop").size().iloc[0]
        fig.suptitle(f"{title}\n(Medians across {n_runs} runs per proportion)",
                    fontsize=16, fontweight="bold", y=0.99)
        fig.subplots_adjust(top=0.82, bottom=0.25, wspace=0.25)

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot to {output_path}")

    def _plot_dual_panel_with_runs(
        self,
        fr_col: str,
        nl_col: str,
        title: str,
        ylabel: str,
        output_path: Path,
        ylim: Optional[tuple[float, float]] = (0, 1),
    ) -> None:
        """Two side-by-side panels (French / Dutch test sentences) with one metric each.
        Plots individual run trajectories, median (thick black dashed), and mean (red dotted)."""
        self.setup_plot_style()
        df = self.results_df.sort_values(["prop", "run_id"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plt.rcParams.update({
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

        def plot_panel(ax, col: str, subtitle: str):
            run_ids = df["run_id"].unique()
            for run_id in run_ids:
                run_df = df[df["run_id"] == run_id].sort_values("prop")
                ax.plot(
                    run_df["prop"],
                    run_df[col],
                    color="gray",
                    alpha=0.4,
                    linewidth=1,
                    zorder=1,
                )
            medians = df.groupby("prop")[col].median()
            means = df.groupby("prop")[col].mean()
            ax.plot(
                medians.index,
                medians.values,
                color="black",
                linewidth=2.5,
                linestyle="--",
                label="Median",
                zorder=3,
            )
            ax.plot(
                means.index,
                means.values,
                color="red",
                linewidth=1.5,
                linestyle=":",
                label="Mean",
                zorder=2,
            )
            ax.set_title(subtitle, fontsize=13, pad=10, fontweight="semibold")
            ax.set_xlabel("Proportion of French in Training Data")
            ax.set_ylabel(ylabel)
            ax.set_ylim(ylim)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper left", fontsize=10)

        plot_panel(ax1, fr_col, "French test sentences")
        plot_panel(ax2, nl_col, "Dutch test sentences")

        n_runs = df.groupby("prop").size().iloc[0] if len(df) else 0
        fig.suptitle(f"{title}\n({n_runs} runs per proportion)", fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot to {output_path}")

    def _plot_dual_panel_multi_series(
        self,
        fr_series: list[tuple[str, str]],
        nl_series: list[tuple[str, str]],
        title: str,
        ylabel: str,
        output_path: Path,
        ylim: Optional[tuple[float, float]] = (0, 1),
    ) -> None:
        """Two side-by-side panels; each panel has multiple series (col, label), each with median + mean."""
        self.setup_plot_style()
        df = self.results_df.sort_values(["prop", "run_id"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plt.rcParams.update({
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        def plot_panel(ax, series_list: list[tuple[str, str]], subtitle: str):
            for idx, (col, label) in enumerate(series_list):
                color = colors[idx % len(colors)]
                medians = df.groupby("prop")[col].median()
                means = df.groupby("prop")[col].mean()
                ax.plot(
                    medians.index,
                    medians.values,
                    color=color,
                    linewidth=2.5,
                    linestyle="--",
                    label=f"{label} (median)",
                    zorder=3,
                )
                ax.plot(
                    means.index,
                    means.values,
                    color=color,
                    linewidth=1.5,
                    linestyle=":",
                    label=f"{label} (mean)",
                    zorder=2,
                )
            ax.set_title(subtitle, fontsize=13, pad=10, fontweight="semibold")
            ax.set_xlabel("Proportion of French in Training Data")
            ax.set_ylabel(ylabel)
            ax.set_ylim(ylim)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper left", fontsize=9)

        plot_panel(ax1, fr_series, "French test sentences")
        plot_panel(ax2, nl_series, "Dutch test sentences")

        n_runs = df.groupby("prop").size().iloc[0] if len(df) else 0
        fig.suptitle(f"{title}\n({n_runs} runs per proportion)", fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot to {output_path}")

    def plot_token_share_by_proportion(self, suffix: Optional[str] = None) -> None:
        """Plot token share (proportion of tokens in expected language) by French training proportion.
        Left: French test sentences (fr_avg_fr). Right: Dutch test sentences (nl_avg_nl).
        If suffix is set (e.g. 'subject'), filename and title include it for ablation-specific plots."""
        base = "token_share_by_proportion"
        fname = f"{base}_{suffix}.png" if suffix else f"{base}.png"
        title = f"Token share (expected language) - {suffix} ablation" if suffix else "Token share (expected language)"
        self._plot_dual_panel_with_runs(
            fr_col="fr_avg_fr",
            nl_col="nl_avg_nl",
            title=title,
            ylabel="Proportion of tokens in expected language",
            output_path=self.output_dir / fname,
            ylim=(0, 1),
        )

    def plot_structure_followed_by_proportion(self, suffix: Optional[str] = None) -> None:
        """Plot follows_fr, follows_nl, and follows_either (sum) by French training proportion.
        Left: French test sentences. Right: Dutch test sentences. Three curves per panel."""
        base = "structure_followed_by_proportion"
        fname = f"{base}_{suffix}.png" if suffix else f"{base}.png"
        title = f"Structure followed (FR / NL / either) - {suffix} ablation" if suffix else "Structure followed (FR / NL / either)"
        self._plot_dual_panel_multi_series(
            fr_series=[
                ("fr_follows_fr", "follows FR"),
                ("fr_follows_nl", "follows NL"),
                ("fr_follows_either", "follows either (sum)"),
            ],
            nl_series=[
                ("nl_follows_fr", "follows FR"),
                ("nl_follows_nl", "follows NL"),
                ("nl_follows_either", "follows either (sum)"),
            ],
            title=title,
            ylabel="Proportion of test sentences",
            output_path=self.output_dir / fname,
            ylim=(0, 1),
        )

    def plot_exact_match(self) -> None:
        """Plot exact match accuracy by French proportion."""
        self._plot_metric_by_proportion(
            fr_metrics=['fr_exact'],
            fr_labels=['french test sentences'],
            nl_metrics=['nl_exact'],
            nl_labels=['dutch test sentences'],
            title=f'Exact Match Accuracy',
            ylabel='Exact Match Accuracy',
            output_path=self.output_dir / "exact_match.png",
            ylim=(0, 1)
        )

    def plot_token_dist(self) -> None:
        """Plot token distribution in French and Dutch outputs.

        This metric analyzes the language mixing in model outputs for French inputs.
        It tracks what proportion of tokens in French outputs come from each language's
        vocabulary:
        - fr_avg_fr: % of tokens from French vocabulary
        - fr_avg_nl: % of tokens from Dutch vocabulary

        This reveals if the model maintains language separation or if it code-switches
        by inserting Dutch tokens into French outputs. High fr_avg_fr and low fr_avg_nl
        indicates good language separation. Increasing fr_avg_nl as Dutch proportion
        increases suggests interference.
        """
        self._plot_metric_by_proportion(
            fr_metrics=['fr_avg_fr', 'fr_avg_nl'],
            fr_labels=['French', 'Dutch'],
            nl_metrics=['nl_avg_nl', 'nl_avg_fr'],
            nl_labels=['Dutch', 'French'],
            title='Token Language Distribution',
            ylabel='Proportion of tokens outputted in either language',
            output_path=self.output_dir / "token_dist.png",
            ylim=(0, 1)
        )


    # plots from sept 15
    def plot_verb_metrics(self) -> None:
        """Plot all verb-related metrics by French proportion.
        
        Shows these metrics:
        - aux_form_correct: Is auxiliary form, its plurality/number, correct (a/heeft vs. ont/hebben)
        - verb_choice_correct: Is participle derived from input verb (do we use gegeten for words with eet)
        - verb_lang_correct: Is auxiliary in correct language
        - aux_matches_verb: Does aux verb match participle language
        """
        self._plot_metric_by_proportion(
            fr_metrics=[
                'fr_verb_lang',          # aux in french (blue)
                'fr_aux_matches_verb',    # aux matches participle lang (orange)
                'fr_aux_form',           # correct aux form (green)
                'fr_verb_choice',         # correct participle choice (grey)
            ],
            fr_labels=[
                'aux in french',
                'aux lang matches participle lang',
                'correct aux form (a/ont)',
                'participle matches input verb (ex. mange -> mangé)'
            ],
            nl_metrics=[
                'nl_verb_lang',          # aux in dutch (blue)
                'nl_aux_matches_verb',    # aux matches participle lang (orange)
                'nl_aux_form',           # correct aux form (green)
                'nl_verb_choice'          # correct participle choice (grey)
            ],
            nl_labels=[
                'aux in dutch',
                'aux lang matches participle lang',
                'correct aux form (heeft/hebben)',
                'participle matches input verb (ex. eet -> gegeten)'
            ],
            title='Auxiliary Verb and Participle Metrics',
            ylabel='Proportion of test sentences where ...',
            output_path=self.output_dir / "verb_metrics.png",
            ylim=(0, 1)
        )

    def plot_determiner_metrics(self) -> None:
        """Plot determiner-related metrics by French proportion.
        
        These metrics analyze the model's handling of determiners:
        - det_lang_correct: Are determiners from correct language
        - det_agreement: Do determiners agree with nouns in number
        
        High values indicate the model has learned:
        - Language-specific determiner systems
        - Number agreement rules for each language
        """
        self._plot_metric_by_proportion(
            fr_metrics=['fr_det_lang', 'fr_det_agreement'],
            fr_labels=['determiner is in French', 'both nouns have correct determiner number (le chien a mangé les chats)'],
            nl_metrics=['nl_det_lang'], # the same determiner in dutch for singular and plural
            nl_labels=['determiner is in Dutch'],
            title='Determiner Metrics',
            ylabel='Proportion of test sentences where ...',
            output_path=self.output_dir / "det_metrics.png",
            ylim=(0, 1)
        )

    # test THIS NOW.
    def plot_ablation_structure_metrics(self) -> None:
        """Plot structure conformity metrics for ablated sentences in the present perfect.
        
        Shows three metrics for each language:
        - Rate of maintaining any valid structure in the conjugated predicted sentence (either French or Dutch)
        - Rate of following French structure specifically
        - Rate of following Dutch structure specifically
        """
        self._plot_metric_by_proportion(
            fr_metrics=['fr_follows_fr', 'fr_follows_nl', 'fr_follows_either', 'fr_part_final'],
            fr_labels=['follows French structure', 'follows Dutch structure', 'follow either structure', 'participle-final'],
            nl_metrics=['nl_follows_fr', 'nl_follows_nl', 'nl_follows_either', 'nl_part_final'],
            nl_labels=['follows French structure', 'follows Dutch structure', 'follow either structure', 'participle-final'],
            title=f'Structure Conformity',
            ylabel='Proportion of test sentences where ...',
            output_path=self.output_dir / "ablation_structure.png",
            ylim=(0, 1)
        )

    def plot_nov11_structure_lexical(self) -> None:
        """
        Plot structure conformity and lexical orientation metrics (Nov 11 analysis).
        """
        self._plot_metric_by_proportion(
            fr_metrics=['fr_follows_fr', 'fr_follows_nl', 'fr_follows_either', 'fr_lexical_score'],
            fr_labels=[
                'follows French structure',
                'follows Dutch structure',
                'follows either structure',
                'lexical orientation (0=dutch, 1=french)'
            ],
            nl_metrics=['nl_follows_fr', 'nl_follows_nl', 'nl_follows_either', 'nl_lexical_score'],
            nl_labels=[
                'follows French structure',
                'follows Dutch structure',
                'follows either structure',
                'lexical orientation (0=dutch, 1=french)'
            ],
            title='Structure Conformity & Lexical Orientation (Nov 11)',
            ylabel='Median across runs',
            output_path=self.output_dir / "nov11_structure_lexical.png",
            ylim=(0, 1)
        )

    def plot_ablation_word_tracking(self) -> None:
        """
        Plot metrics tracking what happens to ablated words.
        Shows whether the model keeps the codeswitched word in its prediction.
        """
        self._plot_metric_by_proportion(
            fr_metrics=['fr_keeps_ablated'],
            fr_labels=['French: keeps codeswitched word'],
            nl_metrics=['nl_keeps_ablated'],
            nl_labels=['Dutch: keeps codeswitched word'],
            title=f'Handling of Codeswitched Words',
            ylabel='Proportion of test sentences',
            output_path=self.output_dir / "ablation_words.png",
            ylim=(0, 1)
        )
    
    def plot_language_orientation(self) -> None:
        """
        Plot language orientation score by French proportion.
        
        Score ranges from 0 (pure Dutch) to 1 (pure French), based on:
        - Sentence structure (FR vs NL word order)
        - Auxiliary verb language
        - Participle language
        - Determiner language
        - Noun language
        """
        self._plot_metric_by_proportion(
            fr_metrics=['fr_orientation'],
            fr_labels=['Language orientation (0=Dutch, 1=French)'],
            nl_metrics=['nl_orientation'],
            nl_labels=['Language orientation (0=Dutch, 1=French)'],
            title='Language Orientation Score',
            ylabel='Orientation Score (0=Dutch, 1=French)',
            output_path=self.output_dir / "orientation.png",
            ylim=(0, 1)
        )
    
    def plot_syntax_score_by_proportion(self, suffix: Optional[str] = None) -> None:
        """Plot syntax score (word order, 0=NL, 1=FR) by French training proportion."""
        base = "syntax_score_by_proportion"
        fname = f"{base}_{suffix}.png" if suffix else f"{base}.png"
        title = f"Syntax score (word order) - {suffix} ablation" if suffix else "Syntax score (word order)"
        self._plot_dual_panel_with_runs(
            fr_col="fr_syntax_score",
            nl_col="nl_syntax_score",
            title=title,
            ylabel="Syntax score (0=NL order, 1=FR order)",
            output_path=self.output_dir / fname,
            ylim=(0, 1),
        )

    def plot_morphology_score_by_proportion(self, suffix: Optional[str] = None) -> None:
        """Plot morphology score (French token share) by French training proportion."""
        base = "morphology_score_by_proportion"
        fname = f"{base}_{suffix}.png" if suffix else f"{base}.png"
        title = f"Morphology score (French token share) - {suffix} ablation" if suffix else "Morphology score (French token share)"
        self._plot_dual_panel_with_runs(
            fr_col="fr_morphology_score",
            nl_col="nl_morphology_score",
            title=title,
            ylabel="Morphology score (0=Dutch, 1=French)",
            output_path=self.output_dir / fname,
            ylim=(0, 1),
        )

    def plot_alignment_score_by_proportion(self, suffix: Optional[str] = None) -> None:
        """Plot alignment score (syntax + morphology average) by French training proportion."""
        base = "alignment_score_by_proportion"
        fname = f"{base}_{suffix}.png" if suffix else f"{base}.png"
        title = f"Alignment (syntax + morphology) - {suffix} ablation" if suffix else "Alignment (syntax + morphology)"
        self._plot_dual_panel_with_runs(
            fr_col="fr_alignment_score",
            nl_col="nl_alignment_score",
            title=title,
            ylabel="Alignment score (0=Dutch, 1=French)",
            output_path=self.output_dir / fname,
            ylim=(0, 1),
        )

    ## TODO: you need to massively clean up what your plots look like.
    def create_all_plots(self) -> None:
        """Create all visualization plots from results."""
        try:
            logger.info("Creating exact match plot...")
            self.plot_exact_match()

            logger.info("Creating token distribution plot...")
            self.plot_token_dist()
            
            logger.info("Creating verb metrics plots...")
            self.plot_verb_metrics()
            
            logger.info("Creating determiner metrics plots...")
            self.plot_determiner_metrics()

            logger.info("Creating ablation structure plots...")
            self.plot_ablation_structure_metrics()
            
            logger.info("Creating ablation word tracking plots...")
            self.plot_ablation_word_tracking()
            
            logger.info("Creating language orientation plot...")
            self.plot_language_orientation()

            logger.info("Creating syntax score plot...")
            self.plot_syntax_score_by_proportion()

            logger.info("Creating morphology score plot...")
            self.plot_morphology_score_by_proportion()

            logger.info("Creating alignment score plot...")
            self.plot_alignment_score_by_proportion()

            logger.info(f"All plots saved to {self.output_dir}")

        except Exception as e: # any exception in one plot call fails all plots.
            logger.error(f"Error creating plots: {e}", exc_info=True)

    ## this prob shouldnt be a method here but whatever
    @classmethod
    def create_plotter_from_run_metrics_dir(
        cls, metrics_dir: Path, plots_subdir: str = "plots", ablation_type: str = "none"
    ) -> "BilingualPlotter":
        """Create plotter from a directory containing experiment run metrics.

        This factory method handles collecting and aggregating metrics from either:
        - A summary.csv file if it exists in the directory
        - Or individual p*_run* directories containing ablation_{ablation_type}_metrics.json (or metrics.json)

        The directory structure should follow the convention:
        metrics_dir/
            summary.csv  (optional)
            p20_run1/
                ablation_none_metrics.json  (or ablation_subject_metrics.json, etc.)
            ...

        Args:
            metrics_dir: Path to directory containing experiment metrics
            plots_subdir: Name of subdirectory to save plots (default: "plots")
            ablation_type: One of "none", "subject", "verb", "object" (default: "none")

        Returns:
            BilingualPlotter instance ready to create visualizations

        Raises:
            ValueError: If no valid results found in the directory
        """
        plots_dir = metrics_dir / plots_subdir
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Try loading summary first
        summary_path = metrics_dir / "summary.csv"
        if summary_path.exists():
            results_df = pd.read_csv(summary_path)
            return cls(results_df, plots_dir)

        # Otherwise collect from individual runs (ablation_{ablation_type}_metrics.json or metrics.json)
        results = []
        metrics_filename = f"ablation_{ablation_type}_metrics.json"
        for run_dir in sorted(metrics_dir.glob("p*_run*")):
            metrics_file = run_dir / metrics_filename
            if not metrics_file.exists():
                metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                logger.warning(f"No metrics found in {run_dir}")
                continue

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Extract run parameters from directory name (pXX.XX_runYY or p20_run1)
            dir_name = run_dir.name
            if dir_name.startswith("p") and "_run" in dir_name:
                parts = dir_name.split("_run")
                try:
                    prop = float(parts[0][1:]) / 100.0
                except ValueError:
                    prop = int(parts[0][1:]) / 100.0
                run_id = int(parts[1])
            else:
                logger.warning(f"Could not parse proportion and run ID from {dir_name}")
                continue

            result = {
                'prop': prop,
                'run_id': run_id,
                'run_dir': str(run_dir),
                **metrics
            }
            results.append(result)

        if not results:
            raise ValueError(f"No valid results found in {metrics_dir}")

        results_df = pd.DataFrame(results).sort_values("prop")
        return cls(results_df, plots_dir)

    def print_metrics_summary(self) -> None:
        """Print summary statistics of available metrics.

        This includes:
        - List of all available metrics columns
        - Basic statistics (mean, std, min, max, etc.) for each metric
        - Number of runs per proportion
        """
        print("\nAvailable metrics:")
        print(self.results_df.columns.tolist())

        print("\nRuns per proportion:")
        runs_per_prop = self.results_df.groupby('prop').size()
        print(runs_per_prop)

        print("\nSummary statistics:")
        # Exclude non-numeric columns from summary
        numeric_cols = self.results_df.select_dtypes(include=['float64', 'int64']).columns
        print(self.results_df[numeric_cols].describe())
