"""
Metrics for evaluating generated sentences in language experiments.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import pandas as pd


class Metrics:
    """Compute various metrics for generated sentences."""

    def __init__(self, lexicon_path: Path):
        """Initialize with lexicon for language-specific metrics.

        Args:
            lexicon_path: Path to lexicon JSON file
        """
        self.lexicon_path = lexicon_path
        self.lexicon = json.load(open(lexicon_path, encoding="utf-8"))

        # Extract word sets for each language
        self.fr_words = self._extract_words("fr")
        self.nl_words = self._extract_words("nl")

        # Pre-compute specific word sets for complex metrics
        self.part_fr = {v["participle"] for v in self.lexicon["VERBS"]["fr"].values()}
        self.part_nl = {v["participle"] for v in self.lexicon["VERBS"]["nl"].values()}
        self.aux_fr = set(self.lexicon["AUX"]["fr"].values())
        self.aux_nl = set(self.lexicon["AUX"]["nl"].values())

    def _extract_words(self, lang: str) -> Set[str]:
        """Extract all words for a given language from the lexicon into a set."""
        words = set()
        for section in self.lexicon.values():
            if lang in section:
                self._add_values_recursive(section[lang], words)
        return words

    def _add_values_recursive(self, obj: Any, words: Set[str]) -> None:
        """Recursively add all string values to word set."""
        if isinstance(obj, str):
            words.add(obj.lower())
        elif isinstance(obj, list):
            for item in obj:
                self._add_values_recursive(item, words)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._add_values_recursive(value, words)

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization for metric computation."""
        return re.findall(r"\w+|[^\s\w]", text.lower())

    def token_lang_frac(self, tokens: List[str]) -> Tuple[float, float]:
        """Calculate fraction of tokens in French/Dutch using simple set membership.

        Args:
            tokens: List of tokens to analyze

        Returns:
            Tuple of (french_fraction, dutch_fraction)
        """
        if not tokens:
            return 0.0, 0.0

        tokens = [t.lower() for t in tokens]
        fr = sum(t in self.fr_words for t in tokens) / len(tokens)
        nl = sum(t in self.nl_words for t in tokens) / len(tokens)
        return fr, nl

    def is_participle_final(self, tokens: List[str], lang: str) -> bool:
        """Check if participle comes after second noun (for tense experiments).

        Args:
            tokens: List of tokens to analyze
            lang: Language code ("fr" or "nl")

        Returns:
            True if participle appears after second noun
        """
        # Get both singular and plural forms from lexicon structure
        nouns = set()
        for word_forms in self.lexicon["NOUNS"][lang].values():
            if isinstance(word_forms, dict):
                nouns.add(word_forms.get("sgl", ""))
                nouns.add(word_forms.get("pl", ""))
            else:
                nouns.add(str(word_forms))

        parts = self.part_fr if lang == "fr" else self.part_nl
        auxes = self.aux_fr if lang == "fr" else self.aux_nl

        idxs_n = [i for i, t in enumerate(tokens) if t in nouns]
        idxs_p = [i for i, t in enumerate(tokens) if t in parts]
        idxs_a = [i for i, t in enumerate(tokens) if t in auxes]

        if len(idxs_n) < 2 or not idxs_p or not idxs_a:
            return False

        return max(idxs_p) > idxs_n[1]

    def exact_match(self, pred: str, gold: str) -> bool:
        """Check if prediction exactly matches gold standard.

        Args:
            pred: Predicted text
            gold: Gold standard text

        Returns:
            True if exact match
        """
        return pred.strip() == gold.strip()

    def compute_all_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute all metrics for a list of predictions.

        Args:
            predictions: List of dicts with keys: 'language', 'prediction', 'gold'

        Returns:
            Dictionary of computed metrics
        """
        records = []

        for pred_dict in predictions:
            lang = pred_dict['language']
            pred = pred_dict['prediction']
            gold = pred_dict['gold']

            tokens = self.tokenize(pred)
            exact = self.exact_match(pred, gold)
            fr, nl = self.token_lang_frac(tokens)
            part_final = self.is_participle_final(tokens, lang)

            records.append({
                'lang': lang,
                'exact': exact,
                'fr_share': fr,
                'nl_share': nl,
                'part_final': part_final
            })

        # Aggregate metrics by language
        df = pd.DataFrame(records)
        metrics = {}

        for lang in df['lang'].unique():
            sub = df[df['lang'] == lang]
            metrics[f"{lang}_exact"] = float(sub['exact'].mean())
            metrics[f"{lang}_avg_fr"] = float(sub['fr_share'].mean())
            metrics[f"{lang}_avg_nl"] = float(sub['nl_share'].mean())
            metrics[f"{lang}_part_final"] = float(sub['part_final'].mean())

        # Overall metrics
        metrics['overall_exact'] = float(df['exact'].mean())

        return metrics

    def save_metrics(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
            output_path: Path to save metrics JSON
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
