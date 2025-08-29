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
    plotter.plot_word_order()
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
                                metric_cols: list[str],
                                metric_labels: list[str],
                                title: str,
                                ylabel: str,
                                output_path: Path,
                                ylim: Optional[tuple[float, float]] = None) -> None:
        """Create line plot showing metric values by French proportion.

        Args:
            metric_cols: List of column names to plot
            metric_labels: Labels for each metric
            title: Plot title
            ylabel: Y-axis label
            output_path: Path to save plot
            ylim: Optional y-axis limits (min, max)
        """
        self.setup_plot_style()

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot lines with confidence intervals for each metric
        colors = ['blue', 'orange', 'green']  # Add more colors if needed
        for i, (col, label) in enumerate(zip(metric_cols, metric_labels)):
            # Create temporary dataframe for this metric
            plot_df = self.results_df[['prop', col]].copy()
            
            # Plot line with shaded confidence interval
            sns.lineplot(
                data=plot_df,
                x='prop',
                y=col,
                label=label,
                ci='sd',
                err_style='band',
                color=colors[i]
            )

            # Calculate and plot mean points
            means_df = self.results_df.groupby('prop')[col].mean()
            plt.plot(means_df.index, means_df.values, 'o', 
                    color=colors[i], markersize=8)

        # Customize plot
        plt.xlabel('French Proportion')
        plt.ylabel(ylabel)
        plt.title(title)
        if ylim:
            plt.ylim(ylim)
        plt.grid(True, alpha=0.3)

        # Add text for number of runs in bottom right
        n_runs = self.results_df.groupby('prop').size().iloc[0]
        plt.text(0.95, 0.02, f'{n_runs} runs per proportion', 
                horizontalalignment='right',
                transform=plt.gca().transAxes,
                fontsize=10)

        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to {output_path}")

    def plot_exact_match(self) -> None:
        """Plot exact match accuracy by French proportion.

        This metric measures the model's ability to perfectly reproduce entire target sequences.
        A high exact match rate indicates the model has learned the complete transformation
        rules for that language. This is the strictest evaluation metric as it requires
        every token to be correct.

        The metric is computed separately for French and Dutch outputs:
        - fr_exact: % of French inputs where output exactly matches target
        - nl_exact: % of Dutch inputs where output exactly matches target
        - overall_exact: % of all inputs where output exactly matches target

        A drop in exact match for one language as the other's proportion increases suggests
        interference between the languages.
        """
        self._plot_metric_by_proportion(
            metric_cols=['overall_exact', 'fr_exact', 'nl_exact'],
            metric_labels=['Overall', 'French', 'Dutch'],
            title='Exact Match Accuracy by French Proportion',
            ylabel='Exact Match Accuracy',
            output_path=self.output_dir / "exact_match.png",
            ylim=(0, 1)
        )

    def plot_word_order(self) -> None:
        """Plot participle position analysis by French proportion.

        This metric analyzes the model's learning of language-specific word order rules.
        French and Dutch differ in their placement of participles:
        - French typically places participles after objects (non-final)
        - Dutch typically places participles at the end (final)

        The metric tracks the rate of participle-final constructions:
        - fr_part_final: % of French outputs with participle in final position
        - nl_part_final: % of Dutch outputs with participle in final position

        This is a key syntactic difference between the languages. The plot reveals if
        the model has learned to maintain distinct word orders for each language or if
        there is interference (e.g., using Dutch word order in French outputs).
        """
        self._plot_metric_by_proportion(
            metric_cols=['fr_part_final', 'nl_part_final'],
            metric_labels=['French participle-final', 'Dutch participle-final'],
            title='Word Order: Participle Position',
            ylabel='Participle-final Rate',
            output_path=self.output_dir / "word_order.png",
            ylim=(0, 1)
        )

    def plot_french_token_dist(self) -> None:
        """Plot token distribution in French outputs.

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
            metric_cols=['fr_avg_fr', 'fr_avg_nl'],
            metric_labels=['French tokens', 'Dutch tokens'],
            title='French Outputs: Token Distribution',
            ylabel='Token Proportion',
            output_path=self.output_dir / "french_token_dist.png",
            ylim=(0, 1)
        )

    def plot_dutch_token_dist(self) -> None:
        """Plot token distribution in Dutch outputs.

        This metric analyzes the language mixing in model outputs for Dutch inputs.
        It tracks what proportion of tokens in Dutch outputs come from each language's
        vocabulary:
        - nl_avg_nl: % of tokens from Dutch vocabulary
        - nl_avg_fr: % of tokens from French vocabulary

        This reveals if the model maintains language separation or if it code-switches
        by inserting French tokens into Dutch outputs. High nl_avg_nl and low nl_avg_fr
        indicates good language separation. Increasing nl_avg_fr as French proportion
        increases suggests interference.
        """
        self._plot_metric_by_proportion(
            metric_cols=['nl_avg_nl', 'nl_avg_fr'],
            metric_labels=['Dutch tokens', 'French tokens'],
            title='Dutch Outputs: Token Distribution',
            ylabel='Token Proportion',
            output_path=self.output_dir / "dutch_token_dist.png",
            ylim=(0, 1)
        )

    def create_all_plots(self) -> None:
        """Create all visualization plots from results.

        This is a convenience method that creates all available plot types in one call.
        Each plot analyzes different aspects of the model's bilingual capabilities based
        on its generated test outputs.
        """
        try:
            logger.info("Creating exact match plot...")
            self.plot_exact_match()

            logger.info("Creating word order plot...")
            self.plot_word_order()

            logger.info("Creating French token distribution plot...")
            self.plot_french_token_dist()

            logger.info("Creating Dutch token distribution plot...")
            self.plot_dutch_token_dist()

            logger.info(f"All plots saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"Error creating plots: {e}", exc_info=True)

    ## this prob shouldnt be a method here but whatever
    @classmethod
    def create_plotter_from_run_metrics_dir(cls, metrics_dir: Path, plots_subdir: str = "plots") -> "BilingualPlotter":
        """Create plotter from a directory containing experiment run metrics.

        This factory method handles collecting and aggregating metrics from either:
        - A summary.csv file if it exists in the directory
        - Or individual p*_run* directories containing metrics.json files

        The directory structure should follow the convention:
        metrics_dir/
            summary.csv  (optional)
            p20_run1/
                metrics.json
            p20_run2/
                metrics.json
            p50_run1/
                metrics.json
            ...

        Args:
            metrics_dir: Path to directory containing experiment metrics
            plots_subdir: Name of subdirectory to save plots (default: "plots")

        Returns:
            BilingualPlotter instance ready to create visualizations

        Example:
            ```python
            plotter = BilingualPlotter.create_plotter_from_run_metrics_dir(Path("results/sweep"))
            plotter.create_all_plots()
            ```

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

        # Otherwise collect from individual runs
        results = []
        for run_dir in metrics_dir.glob("p*_run*"):
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                logger.warning(f"No metrics found in {run_dir}")
                continue

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Extract run parameters from directory name
            dir_name = run_dir.name
            if dir_name.startswith("p") and "_run" in dir_name:
                parts = dir_name.split("_run")
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
