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
        
        # Colors for different metrics (excluding 'overall')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        for i, (col, label) in enumerate(zip(metric_cols, metric_labels)):
            if 'overall' in col:  # Skip overall metrics
                continue
                
            # Plot individual points with transparency
            plt.scatter(self.results_df['prop'], 
                    self.results_df[col],
                    color=colors[i],
                    alpha=0.3,  # Make individual points transparent
                    label=f"{label} (individual runs)")
            
            # Calculate and plot median line
            medians = self.results_df.groupby('prop')[col].median()
            plt.plot(medians.index, 
                    medians.values,
                    color=colors[i],
                    linewidth=2,
                    label=f"{label} (median)")
            
            # Add median points
            plt.scatter(medians.index,
                    medians.values,
                    color=colors[i],
                    s=100,  # Larger size for median points
                    zorder=5)  # Ensure median points are on top
        
        # Customize plot
        plt.xlabel('Proportion of French in Training Data')
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
        
        # Adjust legend to show only median lines
        handles, labels = plt.gca().get_legend_handles_labels()
        median_handles = handles[1::2]  # Take only median entries
        median_labels = labels[1::2]
        plt.legend(median_handles, median_labels)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot to {output_path}")

    def plot_exact_match(self) -> None:
        """Plot exact match accuracy by French proportion."""
        self._plot_metric_by_proportion(
            metric_cols=[
                'overall_exact',
                'fr_exact',
                'nl_exact'
            ],
            metric_labels=['average', 'french test sentences', 'dutch test sentences'],
            title=f'Exact Match Accuracy ({self.ablation_type} ablation)',
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
            metric_labels=['participle-final on French sentences', 'participle-final on Dutch sentences'],
            title='Participle position in present-perfect sentences',
            ylabel='Rate of sentences where participle is in final position',
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
            metric_labels=['French', 'Dutch'],
            title='What language are tokens outputted in on the French test set sentences?',
            ylabel='Proportion of tokens outputted in either language',
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
            metric_labels=['Dutch', 'French'],
            title='What language are tokens outputted in on the Dutch test set sentences?',
            ylabel='Proportion of tokens outputted in either languagen',
            output_path=self.output_dir / "dutch_token_dist.png",
            ylim=(0, 1)
        )

    # plots from sept 15
    def plot_verb_metrics(self) -> None:
        """Plot verb-related metrics by French proportion.
        
        These metrics analyze the model's handling of verbs in perfect tense transformation:
        - verb_lang_correct: Is auxiliary verb in correct language
        - verb_choice_correct: Is the participle derived from input verb
        - aux_form_correct: Is auxiliary verb form correct (a/heeft vs. ont/hebben)
        
        High values indicate the model has learned:
        - Language-specific auxiliary verb systems
        - Correct participle derivation
        - Proper auxiliary verb forms for each language

        Evaluates separately for nl or fr and then combines results
        """
        # Plot French verb metrics
        self._plot_metric_by_proportion(
            metric_cols=['fr_verb_lang', 'fr_verb_choice', 'fr_aux_form'],
            metric_labels=['aux. verb in correct language?', 'participle derived from input verb? ', 'plurality of aux. verb respected?'],
            title='Tracking auxiliary verb and participle correctness on FRENCH test sentences ',
            ylabel='Proportion of test sentences correct',
            output_path=self.output_dir / "french_verb_metrics.png",
            ylim=(0, 1)
        )
        
        # Plot Dutch verb metrics
        self._plot_metric_by_proportion(
            metric_cols=['nl_verb_lang', 'nl_verb_choice', 'nl_aux_form'],
            metric_labels=['aux. verb in correct language?', 'participle derived from input verb? ', 'plurality of aux. verb respected?'],
            title='Tracking auxiliary verb and participle correctness on DUTCH test sentences ',
            ylabel='Proportion of test sentences correct',
            output_path=self.output_dir / "dutch_verb_metrics.png",
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
            metric_cols=['fr_det_lang', 'fr_det_agreement'],
            metric_labels=['determiner in correct language', 'det. number correct?'],
            title='Determiner Metrics on French test sentences',
            ylabel='Proportion of test sentences correct',
            output_path=self.output_dir / "french_det_metrics.png",
            ylim=(0, 1)
        )
        
        self._plot_metric_by_proportion(
            metric_cols=['nl_det_lang', 'nl_det_agreement'],
            metric_labels=['determiner in correct language', 'det. number correct?'],
            title='Determiner Metrics on Dutch test sentences',
            ylabel='Proportion of test sentences correct',
            output_path=self.output_dir / "dutch_det_metrics.png",
            ylim=(0, 1)
        )
    
    ## TODO: add the ablation metrics

    def plot_ablation_structure_metrics(self) -> None:
        """
        Plot structure conformity metrics for ablated sentences.
        TODO: add desc
        """
        # Plot for French test sentences
        self._plot_metric_by_proportion(
            metric_cols=['fr_follows_either'],  # Simplified to basic metric name
            metric_labels=['maintains valid structure'],
            title=f'Structure Maintenance on French Sentences ({self.ablation_type} ablation)',
            ylabel='Rate of maintaining valid structure',
            output_path=self.output_dir / "french_ablation_structure.png",
            ylim=(0, 1)
        )
        
        # Plot for Dutch test sentences
        self._plot_metric_by_proportion(
            metric_cols=['nl_follows_either'],
            metric_labels=['maintains valid structure'],
            title=f'Structure Maintenance on Dutch Sentences ({self.ablation_type} ablation)',
            ylabel='Rate of maintaining valid structure',
            output_path=self.output_dir / "dutch_ablation_structure.png",
            ylim=(0, 1)
        )

    def plot_ablation_word_tracking(self) -> None:
        """
        Plot metrics tracking what happens to ablated words.
        TODO; add desc
        """
        self._plot_metric_by_proportion(
            metric_cols=['fr_keeps_ablated', 'fr_translates_back'], 
            metric_labels=['keep codeswitched word', 'translate back to French'],
            title=f'Handling of Codeswitched Words in French Sentences ({self.ablation_type} ablation)',
            ylabel='Rate of keeping/translating',
            output_path=self.output_dir / "french_ablation_words.png",
            ylim=(0, 1)
        )
        
        self._plot_metric_by_proportion(
            metric_cols=['nl_keeps_ablated', 'nl_translates_back'],  
            metric_labels=['keep codeswitched word', 'translate back to Dutch'],
            title=f'Handling of Codeswitched Words in Dutch Sentences ({self.ablation_type} ablation)',
            ylabel='Rate of keeping/translating',
            output_path=self.output_dir / "dutch_ablation_words.png",
            ylim=(0, 1)
        )

    def plot_aux_verb_consistency(self) -> None:
        """Plot auxiliary verb consistency metrics.
        
        These metrics analyze how auxiliary verbs interact with codeswitching:
        - aux_matches_input: Aux verb stays in input language
        - aux_matches_verb: Aux verb matches participle language
        
        Shows whether the model maintains consistent auxiliary systems
        when parts of the verb phrase are codeswitched.
        """
        self._plot_metric_by_proportion(
            metric_cols=['fr_aux_matches_input', 'fr_aux_matches_verb'],
            metric_labels=['aux matches input lang', 'aux matches participle lang'],
            title=f'Auxiliary Verb Consistency in French Sentences ({self.ablation_type} ablation)',
            ylabel='Rate of consistency',
            output_path=self.output_dir / "french_aux_consistency.png",
            ylim=(0, 1)
        )
        
        self._plot_metric_by_proportion(
            metric_cols=['nl_aux_matches_input', 'nl_aux_matches_verb'],
            metric_labels=['aux matches input lang', 'aux matches participle lang'],
            title=f'Auxiliary Verb Consistency in Dutch Sentences ({self.ablation_type} ablation)',
            ylabel='Rate of consistency',
            output_path=self.output_dir / "dutch_aux_consistency.png",
            ylim=(0, 1)
        )
    
    ## TODO: you need to massively clean up what your plots look like.
    def create_all_plots(self) -> None:
        """Create all visualization plots from results."""
        try:
            logger.info("Creating exact match plot...")
            self.plot_exact_match()

            logger.info("Creating word order plot...")
            self.plot_word_order()

            logger.info("Creating French token distribution plot...")
            self.plot_french_token_dist()

            logger.info("Creating Dutch token distribution plot...")
            self.plot_dutch_token_dist()
            
            # Add new plots
            logger.info("Creating verb metrics plots...")
            self.plot_verb_metrics()
            
            logger.info("Creating determiner metrics plots...")
            self.plot_determiner_metrics()

            logger.info("Creating ablation structure plots...")
            self.plot_ablation_structure_metrics()
            
            logger.info("Creating ablation word tracking plots...")
            self.plot_ablation_word_tracking()
            
            logger.info("Creating auxiliary consistency plots...")
            self.plot_aux_verb_consistency()

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
