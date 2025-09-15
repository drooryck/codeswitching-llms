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

    ###
    # metrics from monday september 15
    ###
    def verb_consistency_metrics(self, input_text: str, pred_text: str, lang: str) -> Dict[str, bool]:
        """Check if the verb in present tense is correctly transformed in perfect tense.
        
        Args:
            input_text: Original present tense sentence
            pred_text: Generated perfect tense sentence
            lang: Language code ("fr" or "nl")
            
        Returns:
            Dictionary with metrics:
            - verb_lang_correct: Is auxiliary verb in correct language
            - verb_choice_correct: Is the participle derived from input verb
            - aux_form_correct: Is auxiliary verb form correct (a/heeft vs. ont/hebben)
        """
        input_tokens = self.tokenize(input_text)
        pred_tokens = self.tokenize(pred_text)
        
        # Get verb from input (assuming it's the first verb found)
        input_verb = None
        verb_info = None
        for token in input_tokens:
            for v in self.lexicon["VERBS"][lang].values():
                if token in v["present"].values():
                    input_verb = token
                    verb_info = v
                    break
            if input_verb:
                break
                
        if not input_verb:
            return {
                "verb_lang_correct": False,
                "verb_choice_correct": False, 
                "aux_form_correct": False
            }
            
        # Find corresponding participle
        expected_participle = verb_info["participle"] 
                
        # Check auxiliary verb
        aux_correct = False
        if lang == "fr":
            aux_correct = any(aux in pred_tokens for aux in ["a", "ont"])
        else:  # nl
            aux_correct = any(aux in pred_tokens for aux in ["heeft", "hebben"])
            
        # Check if participle appears and is correct
        participle_correct = expected_participle in pred_tokens
        
        # Check language of auxiliary
        aux_lang_correct = False
        if lang == "fr":
            aux_lang_correct = any(aux in pred_tokens for aux in self.aux_fr)
        else:
            aux_lang_correct = any(aux in pred_tokens for aux in self.aux_nl)
            
        return {
            "verb_lang_correct": aux_lang_correct,
            "verb_choice_correct": participle_correct,
            "aux_form_correct": aux_correct
        }

    def determiner_metrics(self, text: str, lang: str) -> Dict[str, bool]:
        """Analyze determiner usage and agreement.
        
        Args:
            text: Generated text
            lang: Language code ("fr" or "nl")
            
        Returns:
            Dictionary with metrics:
            - det_lang_correct: Are determiners from correct language
            - det_agreement: Do determiners agree with nouns in number
        """
        tokens = self.tokenize(text)
        
        # Get determiners for each language
        fr_dets = set(self.lexicon["DET"]["fr"].values())
        nl_dets = set(self.lexicon["DET"]["nl"].values())
        
        # Check if determiners are from correct language
        found_dets = [t for t in tokens if t in fr_dets or t in nl_dets]
        det_lang_correct = all(t in (fr_dets if lang == "fr" else nl_dets) 
                            for t in found_dets)
        
        # Check noun-determiner agreement
        # This requires looking at pairs of det+noun and checking number agreement
        agreement_correct = True
        for i, token in enumerate(tokens[:-1]):  # Look at adjacent pairs
            if token in (fr_dets if lang == "fr" else nl_dets):
                next_token = tokens[i+1]
                # Check if next token is noun and get its number
                for noun_forms in self.lexicon["NOUNS"][lang].values():
                    if isinstance(noun_forms, dict):
                        if next_token == noun_forms["sgl"]:
                            # Check if determiner is singular
                            if lang == "fr" and token not in ["le", "la"]:
                                agreement_correct = False
                            elif lang == "nl" and token not in ["de", "het"]:
                                agreement_correct = False
                        elif next_token == noun_forms["pl"]:
                            # Check if determiner is plural
                            if lang == "fr" and token != "les":
                                agreement_correct = False
                            elif lang == "nl" and token != "de":
                                agreement_correct = False
                                
        return {
            "det_lang_correct": det_lang_correct,
            "det_agreement": agreement_correct
        }

    # TODO: these metrics are useless.
    def word_order_metrics(self, text: str, lang: str) -> Dict[str, bool]:
        """Analyze word order patterns specific to each language.
        
        Args:
            text: Generated text
            lang: Language code ("fr" or "nl")
            
        Returns:
            Dictionary with metrics about word order correctness
        """
        # TODO: this is useless rn
        tokens = self.tokenize(text)
        
        if lang == "nl":
            # Check verb-final in subordinate clauses
            # Check verb-second in main clauses
            # Check separable verb particles
            verb_second = False
            verb_final = False
            particle_correct = False
            
            # Implementation details...
            
        else:  # fr
            # Check SVO order
            # Check adjective position (usually after noun)
            # Check negation structure (ne...pas)
            svo_order = False
            adj_position = False
            negation_correct = False
            
            # Implementation details...
        
        return {
            "verb_position": verb_second if lang == "nl" else svo_order,
            "complex_structure": verb_final if lang == "nl" else negation_correct,
            "modifiers": particle_correct if lang == "nl" else adj_position
        }

    def compute_all_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced version of compute_all_metrics with new grammatical metrics."""
        records = []
        
        for pred_dict in predictions:
            lang = pred_dict['language']
            pred = pred_dict['prediction']
            gold = pred_dict['gold']
            input_text = pred_dict['input']
            
            # Basic metrics
            tokens = self.tokenize(pred)
            exact = self.exact_match(pred, gold)
            fr, nl = self.token_lang_frac(tokens)
            part_final = self.is_participle_final(tokens, lang)
            
            # New metrics
            verb_metrics = self.verb_consistency_metrics(input_text, pred, lang)
            det_metrics = self.determiner_metrics(pred, lang)
            # order_metrics = self.word_order_metrics(pred, lang)
            
            records.append({
                'lang': lang,
                'exact': exact,
                'fr_share': fr,
                'nl_share': nl,
                'part_final': part_final,
                **verb_metrics,
                **det_metrics
                # **order_metrics
            })
        
        # Aggregate metrics by language as before
        df = pd.DataFrame(records)
        metrics = {}
        
        for lang in df['lang'].unique():
            sub = df[df['lang'] == lang]
            # Original metrics
            metrics[f"{lang}_exact"] = float(sub['exact'].mean())
            metrics[f"{lang}_avg_fr"] = float(sub['fr_share'].mean())
            metrics[f"{lang}_avg_nl"] = float(sub['nl_share'].mean())
            metrics[f"{lang}_part_final"] = float(sub['part_final'].mean())
            
            # New grammatical metrics
            metrics[f"{lang}_verb_lang"] = float(sub['verb_lang_correct'].mean())
            metrics[f"{lang}_verb_choice"] = float(sub['verb_choice_correct'].mean())
            metrics[f"{lang}_aux_form"] = float(sub['aux_form_correct'].mean())
            metrics[f"{lang}_det_lang"] = float(sub['det_lang_correct'].mean())
            metrics[f"{lang}_det_agreement"] = float(sub['det_agreement'].mean())
            # metrics[f"{lang}_word_order"] = float(sub['verb_position'].mean())
            
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
