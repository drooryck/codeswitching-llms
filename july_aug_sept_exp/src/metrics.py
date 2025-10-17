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
        # TODO: dont init every calculation. also have more of these to save compute.
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
        self.det_sgl = {self.lexicon["DET"]["fr"]["sgl"], self.lexicon["DET"]["nl"]["sgl"]}
        self.det_pl = {self.lexicon["DET"]["fr"]["pl"], self.lexicon["DET"]["nl"]["pl"]}

        # Pre-compute nouns (all forms)
        self.nouns_fr = {form for noun in self.lexicon["NOUNS"]["fr"].values() 
                        for form in [noun["sgl"], noun["pl"]]}
        self.nouns_nl = {form for noun in self.lexicon["NOUNS"]["nl"].values() 
                        for form in [noun["sgl"], noun["pl"]]}

        self.noun_sgl_fr = {noun_data["sgl"] for noun_data in self.lexicon["NOUNS"]["fr"].values()}
        self.noun_pl_fr = {noun_data["pl"] for noun_data in self.lexicon["NOUNS"]["fr"].values()}
        self.noun_sgl_nl = {noun_data["sgl"] for noun_data in self.lexicon["NOUNS"]["nl"].values()}
        self.noun_pl_nl = {noun_data["pl"] for noun_data in self.lexicon["NOUNS"]["nl"].values()}

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

        # Pre-compute verb-to-participle mappings (for present → present perfect conversion)
        self.verb_to_participle_fr = {}
        for verb_data in self.lexicon["VERBS"]["fr"].values():
            participle = verb_data["participle"]
            for present_form in verb_data["present"].values():
                self.verb_to_participle_fr[present_form] = participle
        
        self.verb_to_participle_nl = {}
        for verb_data in self.lexicon["VERBS"]["nl"].values():
            participle = verb_data["participle"]
            for present_form in verb_data["present"].values():
                self.verb_to_participle_nl[present_form] = participle

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
        """Check if the verb in present tense is correctly transformed in perfect tense."""
        input_tokens = self.tokenize(input_text)
        pred_tokens = self.tokenize(pred_text)
        
        # Get input verb from position 2 (assume well-formed input)
        input_verb = input_tokens[2]
        
        # Get subject from position 1 to check singular/plural
        subject = input_tokens[1]
        
        # Get expected participle using existing mapping
        verb_to_participle = self.verb_to_participle_fr if lang == "fr" else self.verb_to_participle_nl
        expected_participle = verb_to_participle.get(input_verb)
        
        if not expected_participle:
            return {
                "verb_lang_correct": False,
                "verb_choice_correct": False, 
                "aux_form_correct": False
            }
        
        # Find all auxiliaries and participles in prediction
        pred_auxes = [t for t in pred_tokens if t in (self.aux_fr | self.aux_nl)]
        pred_parts = [t for t in pred_tokens if t in (self.part_fr | self.part_nl)]
        
        # Check if auxiliary is in correct language
        aux_lang_correct = any(aux in (self.aux_fr if lang == "fr" else self.aux_nl) for aux in pred_auxes)
        
        # Check if participle matches input verb
        participle_correct = expected_participle in pred_parts
        
        # Check auxiliary form (singular vs plural) - hardcoded for now
        aux_form_correct = False
        if lang == "fr":
            # Check if subject is singular or plural, then check corresponding aux form
            if subject in self.noun_sgl_fr:
                aux_form_correct = "a" in pred_auxes  # singular: "a"
            elif subject in self.noun_pl_fr:
                aux_form_correct = "ont" in pred_auxes  # plural: "ont"
        else:  # nl
            if subject in self.noun_sgl_nl:
                aux_form_correct = "heeft" in pred_auxes  # singular: "heeft"
            elif subject in self.noun_pl_nl:
                aux_form_correct = "hebben" in pred_auxes  # plural: "hebben"
        
        return {
            "verb_lang_correct": aux_lang_correct,
            "verb_choice_correct": participle_correct,
            "aux_form_correct": aux_form_correct
        }

    def determiner_metrics(self, text: str, lang: str) -> Dict[str, bool]:
        """Analyze determiner usage and agreement by scanning DET+NOUN bigrams."""
        tokens = self.tokenize(text)
        
        # Get the correct noun sets for this language
        noun_sgl = self.noun_sgl_fr if lang == "fr" else self.noun_sgl_nl
        noun_pl = self.noun_pl_fr if lang == "fr" else self.noun_pl_nl
        
        # --- Language check for found determiners ---
        found_dets = [t for t in tokens if t in self.det_fr or t in self.det_nl]
        correct_lang_dets = self.det_fr if lang == "fr" else self.det_nl
        other_lang_dets = self.det_nl if lang == "fr" else self.det_fr
        
        det_lang_correct = all(t in correct_lang_dets for t in found_dets) and \
                        not any(t in other_lang_dets for t in found_dets)
        
        # --- Check DET+NOUN agreement ---
        agreement_correct = True
        for i in range(len(tokens) - 1):
            det = tokens[i]
            noun = tokens[i + 1]
            
            # Only check if both are in our sets
            if det in (self.det_fr | self.det_nl) and noun in (noun_sgl | noun_pl):
                # Check agreement: both singular OR both plural
                det_sgl = det in self.det_sgl
                det_pl = det in self.det_pl
                noun_sgl_match = noun in noun_sgl
                noun_pl_match = noun in noun_pl
                
                # Agreement fails if determiner and noun don't match in number
                if not ((det_sgl and noun_sgl_match) or (det_pl and noun_pl_match)):
                    agreement_correct = False
                    break
        
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
            token_types[2].startswith('aux') and
            token_types[3].startswith('part') and
            token_types[4].startswith('det') and
            token_types[5].startswith('noun')
        )
        
        # Check Dutch structure: det NOUN aux det NOUN part
        follows_nl = (
            len(token_types) == 6 and
            token_types[0].startswith('det') and
            token_types[1].startswith('noun') and
            token_types[2].startswith('aux') and
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
            input: Input text with ablation (present tense structure: DET NOUN VERB DET NOUN)
            ablation_type: Type of ablation ('subject', 'verb', 'object', 'none')
        
        Returns:
            Dictionary with metrics about the ablated word:
            - keeps_ablated: Whether ablated word appears in prediction

        For edge cases where the subject and object is the same this is wrong, ignoring for now.
        """
        pred_tokens = self.tokenize(pred.lower())
        input_tokens = self.tokenize(input.lower())
        
        # Handle 'none' ablation type - no ablation occurred
        if ablation_type == 'none':
            return {'keeps_ablated': False}
        
        if ablation_type == 'subject':
            # Subject is second token (position 1) in DET NOUN VERB DET NOUN
            input_subj = input_tokens[1]
            keeps_ablated = input_subj in pred_tokens
            
        elif ablation_type == 'verb':
            # Verb is third token (position 2) in DET NOUN VERB DET NOUN
            input_verb = input_tokens[2]
            participle = self.verb_to_participle_fr.get(input_verb) or \
                         self.verb_to_participle_nl.get(input_verb)

            # check for the participle version of this verb
            keeps_ablated = participle in pred_tokens if participle else False
                
        elif ablation_type == 'object':
            # Object is fifth token (position 4) in DET NOUN VERB DET NOUN
            input_obj = input_tokens[4]
            keeps_ablated = input_obj in pred_tokens
        
        else:
            return {'keeps_ablated': False}
        
        return {
            'keeps_ablated': keeps_ablated
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
