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
        self.lexicon_path = lexicon_path
        self.lexicon = json.load(open(lexicon_path, encoding="utf-8"))

        # Extract word sets for each language
        self.fr_words = self._extract_words("fr")
        self.nl_words = self._extract_words("nl")

        # Pre-compute determiners
        self.det_fr = set(form for forms in self.lexicon["DET"]["fr"].values() 
                        for form in ([forms] if isinstance(forms, str) else forms.values()))
        self.det_nl = set(form for forms in self.lexicon["DET"]["nl"].values() 
                        for form in ([forms] if isinstance(forms, str) else forms.values()))

        # Pre-compute nouns (all forms)
        self.nouns_fr = {form for noun in self.lexicon["NOUNS"]["fr"].values() 
                        for form in [noun["sgl"], noun["pl"]]}
        self.nouns_nl = {form for noun in self.lexicon["NOUNS"]["nl"].values() 
                        for form in [noun["sgl"], noun["pl"]]}

        # Pre-compute verb forms (present tense)
        self.verbs_fr = {form for verb in self.lexicon["VERBS"]["fr"].values() 
                        for form in verb["present"].values()}
        self.verbs_nl = {form for verb in self.lexicon["VERBS"]["nl"].values() 
                        for form in verb["present"].values()}

        # Pre-compute participles
        self.part_fr = {v["participle"] for v in self.lexicon["VERBS"]["fr"].values()}
        self.part_nl = {v["participle"] for v in self.lexicon["VERBS"]["nl"].values()}

        # Pre-compute auxiliaries
        self.aux_fr = set(self.lexicon["AUX"]["fr"].values())
        self.aux_nl = set(self.lexicon["AUX"]["nl"].values())

    def _extract_words(self, lang: str) -> Set[str]:
        """Extract all words for a given language from the lexicon into a set."""
        words = set()
        for section in self.lexicon.values():
            if lang in section:
                self._add_values_recursive(section[lang], words)
        return words

    
    # maybe this should go in dataset management
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
        # TODO: dont make this hardcoded
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

    ## TODO: this is a bit long, gpt5
    def determiner_metrics(self, text: str, lang: str) -> Dict[str, bool]:
        """Analyze determiner usage and agreement by scanning DET+NOUN bigrams.

        Args:
            text: Generated text
            lang: Language code ("fr" or "nl")

        Returns:
            {
                "det_lang_correct": bool,  # are all found determiners from the requested language
                "det_agreement": bool      # for every DET+NOUN bigram, does number agree
            }
        """
        tokens = self.tokenize(text)

        # --- 1) determiners & nouns from lexicon (no hardcoding) -----------------
        # determiners by language
        fr_det_sgl = {self.lexicon["DET"]["fr"]["sgl"]}
        fr_det_pl  = {self.lexicon["DET"]["fr"]["pl"]}
        nl_det_sgl = {self.lexicon["DET"]["nl"]["sgl"]}
        nl_det_pl  = {self.lexicon["DET"]["nl"]["pl"]}

        fr_dets_all = fr_det_sgl | fr_det_pl
        nl_dets_all = nl_det_sgl | nl_det_pl

        # which set is "correct" for this call
        correct_lang_dets = fr_dets_all if lang == "fr" else nl_dets_all
        other_lang_dets   = nl_dets_all if lang == "fr" else fr_dets_all

        # map every determiner surface form -> possible numbers ("sgl", "pl")
        # note: a form may map to both (e.g., NL 'de' in your data)
        det_to_numbers: Dict[str, set] = {}
        def add_det(form: str, num: str):
            det_to_numbers.setdefault(form, set()).add(num)

        for form in fr_det_sgl: add_det(form, "sgl")
        for form in fr_det_pl:  add_det(form, "pl")
        for form in nl_det_sgl: add_det(form, "sgl")
        for form in nl_det_pl:  add_det(form, "pl")

        # noun index: surface form -> possible numbers it can represent
        # (handles invariant nouns where sgl == pl)
        noun_to_numbers: Dict[str, set] = {}
        for noun_forms in self.lexicon["NOUNS"][lang].values():
            if not isinstance(noun_forms, dict):
                continue
            sgl_form = noun_forms["sgl"]
            pl_form  = noun_forms["pl"]
            noun_to_numbers.setdefault(sgl_form, set()).add("sgl")
            noun_to_numbers.setdefault(pl_form, set()).add("pl")

        # --- 2) language check for found determiners ------------------------------
        found_dets = [t for t in tokens if t in fr_dets_all or t in nl_dets_all]
        det_lang_correct = all(t in correct_lang_dets for t in found_dets) and \
                        not any(t in other_lang_dets for t in found_dets)

        # --- 3) scan for DET + NOUN bigrams, verify agreement --------------------
        agreement_correct = True
        for i in range(len(tokens) - 1):
            det = tokens[i]
            noun = tokens[i + 1]

            # only evaluate explicit DET+NOUN pairs
            if det not in det_to_numbers or noun not in noun_to_numbers:
                continue

            det_nums  = det_to_numbers[det]     # e.g., {"sgl"} or {"pl"} or {"sgl","pl"}
            noun_nums = noun_to_numbers[noun]   # same idea (invariant nouns -> both)

            # Agreement holds if there is any overlap in intended number(s)
            if det_nums.isdisjoint(noun_nums):
                agreement_correct = False
                # keep scanning to report aggregate result; break early if you prefer
                # break

        return {
            "det_lang_correct": det_lang_correct,
            "det_agreement": agreement_correct
        }


    # TODO: these metrics are useless.
    def ablation_metrics(self, text: str, lang: str) -> Dict[str, bool]:
        """Analyze word order patterns specific to each language.
        
        Args:
            text: Generated text
            lang: Language code ("fr" or "nl")
            
        Returns:
            Dictionary with metrics useful for the ablation studies
            - Rate of correct sentence structure in present perfect for either
              language structure on prediction, regardless of input language
                recall that correct prediction structure is this:
                tgt = (f"{det} {noun} {AUX} {participle} {det} {noun}"
                   if lang=='fr'
                   else f"{det} {noun} {AUX[lang][aux_key]} {det} {noun} {participle}")
            - Rate of correct prediction sentence structure for the given input language
            - Plot/Metric, for each ablation type, does the ablated word
              show up in the prediction sentence
            - Rate of the auxiliary verb being in the other language when the verb in input was in one language
            - 
        """
        return None

    def check_structure_conformity(self, text: str) -> Dict[str, bool]:
        """Check if sentence strictly follows FR or NL structure for present perfect.
        
        FR structure: det NOUN aux part det NOUN
        NL structure: det NOUN aux det NOUN part
        """
        tokens = self.tokenize(text.lower())
        
        # Helper to identify token types
        def get_token_type(token: str) -> str:
            if token in self.det_fr:
                return "det_fr"
            if token in self.det_nl:
                return "det_nl"
            if token in self.nouns_fr:
                return "noun_fr"
            if token in self.nouns_nl:
                return "noun_nl"
            if token in self.aux_fr:
                return "aux_fr"
            if token in self.aux_nl:
                return "aux_nl"
            if token in self.part_fr:
                return "part_fr"
            if token in self.part_nl:
                return "part_nl"
            return "unknown"
        
        # Get type for each token
        token_types = [get_token_type(t) for t in tokens]
        
        # Check French structure: det NOUN aux part det NOUN
        follows_fr = (
            len(token_types) == 6 and
            token_types[0].startswith('det') and
            token_types[1].startswith('noun') and
            token_types[2] == 'aux_fr' and
            token_types[3].startswith('part') and
            token_types[4].startswith('det') and
            token_types[5].startswith('noun')
        )
        
        # Check Dutch structure: det NOUN aux det NOUN part
        follows_nl = (
            len(token_types) == 6 and
            token_types[0].startswith('det') and
            token_types[1].startswith('noun') and
            token_types[2] == 'aux_nl' and
            token_types[3].startswith('det') and
            token_types[4].startswith('noun') and
            token_types[5].startswith('part')
        )
        
        return {
            'follows_fr_structure': follows_fr,
            'follows_nl_structure': follows_nl,
            'follows_either_structure': follows_fr or follows_nl,
            'token_types': token_types  # useful for debugging
        }

    def track_ablated_word(self, pred: str, input: str, ablation_type: str) -> Dict[str, bool]:
        """Track what happened to ablated word in prediction.
        
        Args:
            pred: Predicted text
            input: Input text with ablation
            ablation_type: Type of ablation ('subject', 'verb', 'object')
        
        Returns:
            Dictionary with metrics about the ablated word:
            - keeps_ablated: Whether ablated word appears in prediction
            - translates_back: Whether word was translated back to original language
        """
        pred_tokens = self.tokenize(pred.lower())
        input_tokens = self.tokenize(input.lower())
        
        if ablation_type == 'subject':
            # Subject is second token (after determiner)
            input_subj = input_tokens[1]
            # Check if subject was kept
            keeps_ablated = input_subj in pred_tokens
            # Check if it was translated (appears in either fr or nl nouns)
            in_fr = input_subj in self.nouns_fr
            in_nl = input_subj in self.nouns_nl
            # If input was Dutch and prediction has French, or vice versa
            translates_back = (in_fr and any(t in self.nouns_nl for t in pred_tokens)) or \
                            (in_nl and any(t in self.nouns_fr for t in pred_tokens))
            
        elif ablation_type == 'verb':
            # Find auxiliary and participle in input
            input_aux = next((t for t in input_tokens if t in (self.aux_fr | self.aux_nl)), None)
            input_part = next((t for t in input_tokens if t in (self.part_fr | self.part_nl)), None)
            
            # Check if they're kept in prediction
            keeps_aux = input_aux in pred_tokens if input_aux else False
            keeps_part = input_part in pred_tokens if input_part else False
            keeps_ablated = keeps_aux or keeps_part
            
            # Check if translated back
            if input_aux in self.aux_fr:
                translates_back = any(t in self.aux_nl for t in pred_tokens)
            elif input_aux in self.aux_nl:
                translates_back = any(t in self.aux_fr for t in pred_tokens)
            else:
                translates_back = False
                
        elif ablation_type == 'object':
            # Object is last token
            input_obj = input_tokens[-1]
            # Check if object was kept
            keeps_ablated = input_obj in pred_tokens
            # Check if it was translated
            in_fr = input_obj in self.nouns_fr
            in_nl = input_obj in self.nouns_nl
            translates_back = (in_fr and any(t in self.nouns_nl for t in pred_tokens)) or \
                            (in_nl and any(t in self.nouns_fr for t in pred_tokens))
        
        else:
            return {'keeps_ablated': False, 'translates_back': False}
        
        return {
            'keeps_ablated': keeps_ablated,
            'translates_back': translates_back
        }

    def check_aux_verb_consistency(self, pred: str, input: str, input_lang: str) -> Dict[str, bool]:
        """Check auxiliary verb language consistency with input verb.
        
        Args:
            pred: Predicted text
            input: Input text
            input_lang: Original language of input ('fr' or 'nl')
        
        Returns:
            Dictionary with auxiliary verb metrics:
            - aux_matches_input: Whether aux verb is in same language as input
            - aux_matches_verb: Whether aux verb matches participle language
        """
        pred_tokens = self.tokenize(pred.lower())
        
        # Find auxiliary and participle in prediction
        pred_aux = next((t for t in pred_tokens if t in (self.aux_fr | self.aux_nl)), None)
        pred_part = next((t for t in pred_tokens if t in (self.part_fr | self.part_nl)), None)
        
        if not (pred_aux and pred_part):
            return {
                'aux_matches_input': False,
                'aux_matches_verb': False
            }
        
        # Check if auxiliary matches input language
        aux_matches_input = (pred_aux in self.aux_fr and input_lang == 'fr') or \
                        (pred_aux in self.aux_nl and input_lang == 'nl')
        
        # Check if auxiliary matches participle language
        aux_in_fr = pred_aux in self.aux_fr
        part_in_fr = pred_part in self.part_fr
        aux_matches_verb = aux_in_fr == part_in_fr
        
        return {
            'aux_matches_input': aux_matches_input,
            'aux_matches_verb': aux_matches_verb
        }
        
    def compute_all_metrics(self, predictions: List[Dict[str, Any]], ablation_type: str = 'none') -> Dict[str, Any]:
        """
        Compute all metrics including structure and ablation metrics regardless of type.
        Metrics are saved to separate files by ablation type, so we don't need to include
        the ablation type in the metric names themselves.
        Ablation type only used for track_ablated_word
        """
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
            
            # Structure metrics
            structure = self.check_structure_conformity(pred)
            
            # Verb and determiner metrics
            verb_metrics = self.verb_consistency_metrics(input_text, pred, lang)
            det_metrics = self.determiner_metrics(pred, lang)
            
            # Ablation and auxiliary metrics (compute for all types)
            ablation = self.track_ablated_word(pred, input_text, ablation_type)
            aux_metrics = self.check_aux_verb_consistency(pred, input_text, lang)
            
            records.append({
                'lang': lang,
                'exact': exact,
                'fr_share': fr,
                'nl_share': nl,
                'part_final': part_final,
                **structure,
                **verb_metrics,
                **det_metrics,
                **ablation,
                **aux_metrics
            })
        
        # Aggregate metrics by language
        df = pd.DataFrame(records)
        metrics = {}
        
        for lang in df['lang'].unique():
            sub = df[df['lang'] == lang]
            prefix = lang  # Just use language as prefix, no ablation type needed
            
            # Standard metrics
            metrics[f"{prefix}_exact"] = float(sub['exact'].mean())
            metrics[f"{prefix}_avg_fr"] = float(sub['fr_share'].mean())
            metrics[f"{prefix}_avg_nl"] = float(sub['nl_share'].mean())
            metrics[f"{prefix}_part_final"] = float(sub['part_final'].mean())
            
            # Verb and determiner metrics
            metrics[f"{prefix}_verb_lang"] = float(sub['verb_lang_correct'].mean())
            metrics[f"{prefix}_verb_choice"] = float(sub['verb_choice_correct'].mean())
            metrics[f"{prefix}_aux_form"] = float(sub['aux_form_correct'].mean())
            metrics[f"{prefix}_det_lang"] = float(sub['det_lang_correct'].mean())
            metrics[f"{prefix}_det_agreement"] = float(sub['det_agreement'].mean())
            
            # Structure metrics
            metrics[f"{prefix}_follows_fr"] = float(sub['follows_fr_structure'].mean())
            metrics[f"{prefix}_follows_nl"] = float(sub['follows_nl_structure'].mean())
            metrics[f"{prefix}_follows_either"] = float(sub['follows_either_structure'].mean())
            
            # Ablation metrics
            metrics[f"{prefix}_keeps_ablated"] = float(sub['keeps_ablated'].mean())
            metrics[f"{prefix}_translates_back"] = float(sub['translates_back'].mean())
            metrics[f"{prefix}_aux_matches_input"] = float(sub['aux_matches_input'].mean())
            metrics[f"{prefix}_aux_matches_verb"] = float(sub['aux_matches_verb'].mean())
        
        # Overall metrics (no ablation type needed in names)
        metrics['overall_exact'] = float(df['exact'].mean())
        metrics['overall_follows_structure'] = float(df['follows_either_structure'].mean())
        metrics['overall_keeps_ablated'] = float(df['keeps_ablated'].mean())
        
        return metrics
    

    def save_metrics(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
            output_path: Path to save metrics JSON
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
