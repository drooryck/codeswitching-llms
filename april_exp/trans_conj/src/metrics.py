"""
Metrics for evaluating generated sentences in language experiments.
"""
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import pandas as pd


logger = logging.getLogger(__name__)


class Metrics:
    """Compute various metrics for generated sentences."""

    TOKEN_PATTERN = re.compile(r"<[^>\s]+>|\w+|[^\s\w]")
    SPECIAL_TOKEN_PATTERN = re.compile(r"<[^>\s]+>")
    FR_TEMPLATE_456 = ("part", "det", "noun")

    FR_POS_TEMPLATE = ("det", "noun", "aux", "part", "det", "noun")
    NL_POS_TEMPLATE = ("det", "noun", "aux", "det", "noun", "part")
    VALID_POS_PER_POSITION = [
        frozenset({fr, nl})
        for fr, nl in zip(FR_POS_TEMPLATE, NL_POS_TEMPLATE)
    ]

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

        # Expected token length for present perfect template:
        # DET NOUN AUX PART DET NOUN
        self.expected_token_len = 6

        # Track which tokens we've already warned about during lexical scoring
        self._lexical_warning_tokens: Set[str] = set()

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
        return self.TOKEN_PATTERN.findall(text.lower())

    def lexical_mixture_score(self, tokens: List[str]) -> float:
        """
        Measure how French-leaning the sentence is based on token lexicon membership.

        Returns:
            Score in [0, 1] where 1 means all tokens are uniquely French,
            0 means all tokens are uniquely Dutch, and 0.5 represents ambiguous/OOV tokens.
        """
        if not tokens:
            return 0.0

        score = 0.0
        valid_count = 0
        for tok in tokens:
            if self.SPECIAL_TOKEN_PATTERN.fullmatch(tok):
                # Treat special tokens (e.g., <pad>) as neutral; skip them entirely.
                continue

            is_fr = tok in self.fr_words
            is_nl = tok in self.nl_words

            if is_fr and not is_nl:
                score += 1.0
                valid_count += 1
            elif is_nl and not is_fr:
                score += 0.0
                valid_count += 1
            elif is_fr and is_nl:
                # Token exists in both lexicons; treat as neutral but keep in denominator.
                self._log_lexical_warning(tok, is_fr, is_nl)
                score += 0.5
                valid_count += 1
            else:
                # Unknown token; skip it entirely (does not affect score or count).
                self._log_lexical_warning(tok, is_fr, is_nl)

        if valid_count == 0:
            return 0.0

        return score / valid_count

    def _log_lexical_warning(self, token: str, is_fr: bool, is_nl: bool) -> None:
        """Log a warning once per token when it doesn't map cleanly to either lexicon."""
        if token in self._lexical_warning_tokens:
            return
        self._lexical_warning_tokens.add(token)

        if is_fr and is_nl:
            logger.warning(
                "Lexical metric: token '%s' appears in both FR and NL lexicons; treating as neutral (0.5).",
                token,
            )
        else:
            logger.warning(
                "Lexical metric: token '%s' not found in FR/NL lexicons; treating as neutral (0.5).",
                token,
            )

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

    ## get language orientation score over prediction sentences
    def language_orientation_score(
        self,
        pred: str,
        ablation_type: str | None = None,
        input_lang: str | None = None,
        return_breakdown: bool = False
    ):
        """
        Frenchiness in [0,1]: 0 = strongly Dutch, 1 = strongly French.
        Weights: structure(0.35) > aux(0.30) > participle(0.20) > det(0.10) > noun(0.05).
        Missing/unrecognized cues contribute 0.5 (neutral).
        
        When return_breakdown=True, also returns quality metrics (no weight in score):
        - structure_bad: 1.0 if neither FR nor NL structure, 0.0 otherwise
        - aux_missing: 1.0 if no auxiliary found
        - participle_missing: 1.0 if no participle found
        - determiners_missing: 1.0 if no determiners found
        - nouns_missing: 1.0 if no nouns found
        - wrong_token_count: 1.0 if not exactly 6 tokens (expected structure)
        - wrong_noun_count: 1.0 if not exactly 2 nouns (duplicate/missing nouns)
        - wrong_det_count: 1.0 if not exactly 2 determiners
        - wrong_aux_count: 1.0 if not exactly 1 auxiliary
        - wrong_part_count: 1.0 if not exactly 1 participle
        """
        NEU = 0.5
        w_struct, w_aux, w_part, w_det, w_noun = 0.2, 0.2, 0.20, 0.20, 0.2

        tokens = self.tokenize(pred.lower())
        
        # Count token types for quality checks
        n_tokens = len(tokens)
        n_nouns = len([t for t in tokens if t in (self.nouns_fr | self.nouns_nl)])
        n_dets = len([t for t in tokens if t in (self.det_fr | self.det_nl)])
        n_auxes = len([t for t in tokens if t in (self.aux_fr | self.aux_nl)])
        n_parts = len([t for t in tokens if t in (self.part_fr | self.part_nl)])

        # --- Structure: FR=1, NL=0, neither=0.5
        st = self.check_structure_conformity(pred)
        if   st.get("follows_fr_structure"): 
            s_struct = 1.0
            s_struct_bad = 0.0
        elif st.get("follows_nl_structure"): 
            s_struct = 0.0
            s_struct_bad = 0.0
        else:
            s_struct = NEU
            s_struct_bad = 1.0  # Bad: follows neither structure

        # --- Aux: FR=1, NL=0, missing=0.5
        aux = next((t for t in tokens if t in (self.aux_fr | self.aux_nl)), None)
        aux_missing = 1.0 if aux is None else 0.0
        if   aux is None:           s_aux = NEU
        elif aux in self.aux_fr:    s_aux = 1.0
        elif aux in self.aux_nl:    s_aux = 0.0
        else:                       s_aux = NEU

        # --- Participle: FR=1, NL=0, missing=0.5
        part = next((t for t in tokens if t in (self.part_fr | self.part_nl)), None)
        part_missing = 1.0 if part is None else 0.0
        if   part is None:          s_part = NEU
        elif part in self.part_fr:  s_part = 1.0
        elif part in self.part_nl:  s_part = 0.0
        else:                       s_part = NEU

        # --- Determiners: fraction FR among recognized dets; none -> 0.5
        dets = [t for t in tokens if t in (self.det_fr | self.det_nl)]
        dets_missing = 1.0 if not dets else 0.0
        if dets:
            fr = sum(t in self.det_fr for t in dets)
            nl = sum(t in self.det_nl for t in dets)
            s_det = (fr / (fr + nl)) if (fr + nl) > 0 else NEU
        else:
            s_det = NEU

        # --- Nouns: fraction FR among recognized nouns; none -> 0.5
        nouns = [t for t in tokens if t in (self.nouns_fr | self.nouns_nl)]
        nouns_missing = 1.0 if not nouns else 0.0
        if nouns:
            fr = sum(t in self.nouns_fr for t in nouns)
            nl = sum(t in self.nouns_nl for t in nouns)
            s_noun = (fr / (fr + nl)) if (fr + nl) > 0 else NEU
        else:
            s_noun = NEU
            
        score = (
            w_struct * s_struct +
            w_aux    * s_aux +
            w_part   * s_part +
            w_det    * s_det +
            w_noun   * s_noun
        )

        if return_breakdown:
            return {
                'score': float(max(0.0, min(1.0, score))),
                'components': {
                    'structure': s_struct,
                    'aux': s_aux,
                    'participle': s_part,
                    'determiner': s_det,
                    'noun': s_noun
                },
                'quality': {
                    'structure_bad': s_struct_bad,
                    'aux_missing': aux_missing,
                    'participle_missing': part_missing,
                    'determiners_missing': dets_missing,
                    'nouns_missing': nouns_missing,
                    'wrong_token_count': 1.0 if n_tokens != 6 else 0.0,
                    'wrong_noun_count': 1.0 if n_nouns != 2 else 0.0,
                    'wrong_det_count': 1.0 if n_dets != 2 else 0.0,
                    'wrong_aux_count': 1.0 if n_auxes != 1 else 0.0,
                    'wrong_part_count': 1.0 if n_parts != 1 else 0.0
                }
            }

        return float(max(0.0, min(1.0, score)))
            
    def _get_token_role(self, token: str) -> str:
        """Classify a token into its grammatical role."""
        t = token.lower()
        if t in self.det_fr or t in self.det_nl:
            return "det"
        if t in self.nouns_fr or t in self.nouns_nl:
            return "noun"
        if t in self.aux_fr or t in self.aux_nl:
            return "aux"
        if t in self.part_fr or t in self.part_nl:
            return "part"
        return "unknown"

    FR_TEMPLATE = ("det", "noun", "aux", "part", "det", "noun")
    NL_TEMPLATE = ("det", "noun", "aux", "det", "noun", "part")

    def per_position_pos_validity(self, prediction: str) -> Optional[float]:
        """Fraction of the 6 token positions whose POS matches either template.

        Returns None when the prediction does not contain exactly 6 tokens.
        """
        tokens = self.tokenize(prediction)
        if len(tokens) != 6:
            return None
        hits = sum(
            self._get_token_role(t) in self.VALID_POS_PER_POSITION[i]
            for i, t in enumerate(tokens)
        )
        return hits / 6.0

    def pos_coverage_rate(self, prediction: str) -> float:
        """Position-aware POS coverage over the output.

        For each of the first 6 positions, checks whether the token's POS
        matches what either the FR or NL template expects at that position.
        Tokens beyond position 5 always score 0. The denominator is
        max(len(tokens), 6) so excessively long outputs are penalised.

        Returns a value in [0, 1]. Never returns None.
        """
        tokens = self.tokenize(prediction)
        if not tokens:
            return 0.0
        denom = max(len(tokens), 6)
        hits = 0
        for i, t in enumerate(tokens):
            if i >= 6:
                break
            if self._get_token_role(t) in self.VALID_POS_PER_POSITION[i]:
                hits += 1
        return hits / denom

    def compute_syntax_score(self, prediction: str) -> Optional[dict]:
        """Syntax score (0=NL word order, 1=FR word order) from divergent positions 3-5.

        Only counts positions where the token matches one of the two expected
        roles (FR or NL). Positions matching neither are excluded, giving a
        cleaner signal of syntactic preference. Pair with conformity_score for
        overall well-formedness.

        Returns None if prediction doesn't have exactly 6 tokens or positions
        0-2 don't follow the shared prefix (det noun aux).
        Otherwise returns dict with score, valid_positions, total_divergent.
        """
        tokens = self.tokenize(prediction.lower())
        if len(tokens) != 6:
            return None
        roles = [self._get_token_role(t) for t in tokens]
        if roles[0] != "det" or roles[1] != "noun" or roles[2] != "aux":
            return None

        total = 0.0
        valid = 0
        for i, fr_role in enumerate(self.FR_TEMPLATE_456):
            nl_role = self.NL_TEMPLATE[3 + i]
            pred_role = roles[3 + i]
            if pred_role == fr_role:
                total += 1.0
                valid += 1
            elif pred_role == nl_role:
                valid += 1
        score = total / valid if valid > 0 else None
        return {"score": score, "valid_positions": valid, "total_divergent": 3}

    def compute_conformity_score(self, prediction: str) -> dict:
        """Fraction of output tokens conforming to EITHER language's grammar.

        For each position i:
          - If i < 6: 1 if role matches FR_TEMPLATE[i] or NL_TEMPLATE[i], else 0.
          - If i >= 6: 0 (extra tokens are always wrong).
        Denominator is max(len(output), 6).

        Returns dict with score, matches, denominator.
        """
        tokens = self.tokenize(prediction.lower())
        template_len = 6

        matches = 0
        for i, tok in enumerate(tokens):
            if i >= template_len:
                break
            role = self._get_token_role(tok)
            if role == self.FR_TEMPLATE[i] or role == self.NL_TEMPLATE[i]:
                matches += 1

        denom = max(len(tokens), template_len)
        return {"score": matches / denom if denom > 0 else 0.0,
                "matches": matches, "denominator": denom}

    def compute_lexical_score(self, prediction: str) -> float:
        """Fraction of tokens that are French (lexical language membership)."""
        tokens = self.tokenize(prediction.lower())
        if not tokens:
            return 0.0
        fr_frac, _ = self.token_lang_frac(tokens)
        return fr_frac

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
            token_count = len(tokens)
            tokens_expected_len = int(token_count == self.expected_token_len)
            lexical_score = self.lexical_mixture_score(tokens)
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
            
            # Language orientation score
            orientation = self.language_orientation_score(pred)

            # Alignment metrics (syntax = word order, lexical = token language)
            syntax_result = self.compute_syntax_score(pred)
            syntax_score = syntax_result["score"] if syntax_result is not None else None
            syntax_valid_positions = syntax_result["valid_positions"] if syntax_result is not None else 0
            conformity = self.compute_conformity_score(pred)
            lexical_fr_score = self.compute_lexical_score(pred)
            alignment_score = ((syntax_score + lexical_fr_score) / 2.0) if syntax_score is not None else None

            pos_validity = self.per_position_pos_validity(pred)
            pos_coverage = self.pos_coverage_rate(pred)

            records.append({
                'lang': lang,
                'exact': exact,
                'fr_share': fr,
                'nl_share': nl,
                'part_final': part_final,
                'orientation_score': orientation,
                'lexical_mixture_score': lexical_score,
                'tokens_expected_len': tokens_expected_len,
                'syntax_score': syntax_score,
                'syntax_valid_positions': syntax_valid_positions,
                'conformity_score': conformity["score"],
                'lexical_score': lexical_fr_score,
                'alignment_score': alignment_score,
                'pos_validity': pos_validity,
                'pos_coverage': pos_coverage,
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
            prefix = lang 
            
            # Standard metrics
            metrics[f"{prefix}_exact"] = float(sub['exact'].mean())
            metrics[f"{prefix}_avg_fr"] = float(sub['fr_share'].mean())
            metrics[f"{prefix}_avg_nl"] = float(sub['nl_share'].mean())
            metrics[f"{prefix}_part_final"] = float(sub['part_final'].mean())
            metrics[f"{prefix}_orientation"] = float(sub['orientation_score'].mean())
            metrics[f"{prefix}_lexical_mixture_score"] = float(sub['lexical_mixture_score'].mean())
            metrics[f"{prefix}_pct_expected_len"] = float(sub['tokens_expected_len'].mean())
            
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

            # Alignment metrics
            valid_syntax = sub['syntax_score'].dropna()
            metrics[f"{prefix}_syntax_score"] = float(valid_syntax.mean()) if len(valid_syntax) else None
            metrics[f"{prefix}_conformity_score"] = float(sub['conformity_score'].mean())
            metrics[f"{prefix}_lexical_score"] = float(sub['lexical_score'].mean())
            valid_align = sub['alignment_score'].dropna()
            metrics[f"{prefix}_alignment_score"] = float(valid_align.mean()) if len(valid_align) else None
            metrics[f"{prefix}_syntax_valid_pct"] = float(len(valid_syntax) / len(sub)) if len(sub) else 0.0

            valid_pos_val = sub['pos_validity'].dropna()
            metrics[f"{prefix}_pos_validity"] = float(valid_pos_val.mean()) if len(valid_pos_val) else None
            metrics[f"{prefix}_pos_coverage"] = float(sub['pos_coverage'].mean())
        
        # Overall metrics (no ablation type needed in names)
        metrics['overall_exact'] = float(df['exact'].mean())
        metrics['overall_follows_structure'] = float(df['follows_either_structure'].mean())
        metrics['overall_keeps_ablated'] = float(df['keeps_ablated'].mean())
        metrics['overall_lexical'] = float(df['lexical_score'].mean())
        metrics['overall_lexical_mixture'] = float(df['lexical_mixture_score'].mean())
        metrics['overall_conformity'] = float(df['conformity_score'].mean())
        metrics['overall_pct_expected_len'] = float(df['tokens_expected_len'].mean())
        
        return metrics
    

    def save_metrics(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
            output_path: Path to save metrics JSON
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
