"""
Dataset management for language experiments.
"""
import json
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece
from transformers import PreTrainedTokenizerFast
from .model_config import ModelConfig
import re
import itertools


class DatasetManager:
    """Manage datasets, lexicons, and tokenizers for language experiments."""

    def __init__(self, data_dir: str, config: ModelConfig, lexicon_path: str):
        """Initialize with data directory and model config.

        Args:
            data_dir: Directory containing data files
            config: Model configuration containing tokenizer settings
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.tokenizer = None
        self.lexicon = None
        self.datasets = {}
        self.lexicon_path = Path(lexicon_path)

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def tokenizer_config(self) -> Dict[str, str]:
        """Get tokenizer configuration from model config."""
        if self.config is None:
            raise ValueError("ModelConfig is required for tokenizer configuration")
        return self.config.tokenizer_config

    def load_lexicon(self) -> dict:
        """Load lexicon from JSON file"""
        if self.lexicon is None:
            with open(self.lexicon_path, 'r') as f:
                self.lexicon = json.load(f)
        return self.lexicon

    def save_lexicon(self, lexicon: Dict[str, Any], path: Optional[Path] = None) -> None:
        """Save lexicon to file.

        Args:
            lexicon: Lexicon dictionary to save
            path: Path to save lexicon (defaults to data_dir/lexicon_new.json)
        """
        if path is None:
            path = self.lexicon_path

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(lexicon, f, ensure_ascii=False, indent=2)

        self.lexicon = lexicon

    def extract_words(self, lang: str, lexicon: Optional[Dict] = None) -> Set[str]:
        """Extract all words for a given language from the lexicon into a set.

        Args:
            lang: Language code ("fr" or "nl")
            lexicon: Lexicon to use (defaults to self.lexicon)

        Returns:
            Set of all words for the language
        """
        if lexicon is None:
            lexicon = self.lexicon

        if lexicon is None:
            raise ValueError("No lexicon loaded. Call load_lexicon() first.")

        words = set()
        for section in lexicon.values():
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

    # TODO: is this dated method
    def load_vocab(self, path: Optional[Path] = None) -> List[str]:
        """Load vocabulary.

        Args:
            path: Path to vocab file (defaults to data_dir/vocab.json)

        Returns:
            List of vocabulary tokens
        """
        if path is None:
            path = self.data_dir / "vocab.json"

        self.vocab = json.load(open(path, encoding="utf-8"))
        return self.vocab

    def build_tokenizer(self) -> PreTrainedTokenizerFast:
        """Build tokenizer from vocabulary"""
        if self.tokenizer is not None:
            return self.tokenizer

        # Load datasets to build vocabulary
        train_df = pd.read_csv(self.data_dir / "train.csv")
        test_df = pd.read_csv(self.data_dir / "test.csv")

        # Build vocabulary
        def tok(s):
            return re.findall(r"\w+|[^\s\w]", s)

        vocab = set()
        for df in [train_df, test_df]:
            for col in ['input', 'target']:
                for text in df[col]:
                    vocab.update(tok(text))

        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>', '<sep>']
        vocab = special_tokens + sorted(vocab)

        # Create tokenizer
        tokenizer = Tokenizer(WordLevel(vocab={w: i for i, w in enumerate(vocab)},
                                       unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = WordPiece()

        # Wrap with HuggingFace tokenizer
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            **self.tokenizer_config
        )

        return self.tokenizer

    def make_and_save_testing_and_training_data(self, test_size: float = 0.2, random_seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create and save train/test splits with complex subject-verb pair splitting.

        This method implements the sophisticated data splitting logic where training and testing sets
        are split to ensure (language, subject, verb) combinations don't appear in both sets.
        This prevents the model from memorizing specific subject-verb pairs and forces it to generalize.
        The resulting datasets are saved to new_train.csv and new_test.csv.

        Args:
            test_size: Proportion of data to use for testing (default 0.2)
            random_seed: Random seed for reproducible splits (default 0)

        Returns:
            tuple: (train_df, test_df) pandas DataFrames
        """
        lexicon = self.load_lexicon()

        # Extract lexicon components
        DET = lexicon["DET"]
        NOUNS = lexicon["NOUNS"]
        VERBS = lexicon["VERBS"]
        AUX = lexicon["AUX"]

        def get_verb_form(verb_info: dict, lang: str, plural: bool) -> str:
            """Get the appropriate verb form based on language and plurality."""
            if lang == "nl":
                return verb_info["present"]["zijpl" if plural else "hij"]
            else:  # fr
                return verb_info["present"]["ils" if plural else "il"]

        def make_pair(lang, subj, obj, verb_key, plural):
            det = DET[lang]["pl" if plural else "sgl"]
            s = NOUNS[lang][subj]["pl" if plural else "sgl"]
            o = NOUNS[lang][obj]["pl" if plural else "sgl"]

            # Get verb forms
            verb_info = VERBS[lang][verb_key]
            pres = get_verb_form(verb_info, lang, plural)
            part = verb_info["participle"]

            # important couple lines
            aux_key = ("zijpl" if plural else "hij") if lang == "nl" else ("ils" if plural else "il")
            inp = f"{det} {s} {pres} {det} {o}"
            tgt = (f"{det} {s} {AUX[lang][aux_key]} {part} {det} {o}"
                   if lang=='fr'
                   else f"{det} {s} {AUX[lang][aux_key]} {det} {o} {part}")
            return inp, tgt

        rows = []
        pair_ids = []  # keep (lang, subj, verb) for later split

        for lang in ('fr', 'nl'):
            nouns = list(NOUNS[lang])
            verbs = list(VERBS[lang])
            for subj, obj, verb_key in itertools.product(nouns, nouns, verbs):
                if subj == obj:
                    continue
                for plural in (False, True):
                    inp, tgt = make_pair(lang, subj, obj, verb_key, plural)
                    rows.append({
                        'input': inp,
                        'target': tgt,
                        'lang': lang,
                        'plural': plural,
                        'subj': subj,
                        'obj': obj,
                        'verb': verb_key
                    })
                    pair_ids.append((lang, subj, verb_key))  # key for split

        # Split by unseen (subject, verb) combinations
        import random
        pair_ids_unique = list(set(pair_ids))
        random.seed(random_seed)
        test_pairs = set(random.sample(pair_ids_unique,
                                     int(test_size * len(pair_ids_unique))))

        train_rows, test_rows = [], []
        for row, pid in zip(rows, pair_ids):
            (test_rows if pid in test_pairs else train_rows).append(row)

        train_df = pd.DataFrame(train_rows).reset_index(drop=True)
        test_df = pd.DataFrame(test_rows).reset_index(drop=True)

        # Save datasets - TODO: this shouldnt be hardcoded to new_train or new_test
        train_df.to_csv(self.data_dir / "train.csv", index=False)
        test_df.to_csv(self.data_dir / "test.csv", index=False)

        return train_df, test_df

    def create_pytorch_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       tokenizer: Optional[PreTrainedTokenizerFast] = None) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Create PyTorch datasets from dataframes.

        Args:
            train_df: Training dataframe
            test_df: Testing dataframe
            tokenizer: Optional tokenizer (will build if not provided)

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        if tokenizer is None:
            raise ValueError("No tokenizer available. Call build_tokenizer() first.")

        def encode_pair(pres: str, past: str) -> Dict[str, List[int]]:
            """Encode input-target pair for sequence-to-sequence training."""
            pres_ids = tokenizer.encode(pres, add_special_tokens=False)
            past_ids = tokenizer.encode(past, add_special_tokens=False)
            ids = [tokenizer.bos_token_id] + pres_ids + [tokenizer.sep_token_id] + past_ids + [tokenizer.eos_token_id]
            sep_pos = 1 + len(pres_ids)
            labels = [-100] * (sep_pos + 1) + ids[sep_pos + 1:]
            if len(labels) < len(ids):
                labels += [-100] * (len(ids) - len(labels))
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": labels
            }

        class PairDataset(torch.utils.data.Dataset):
            def __init__(self, df: pd.DataFrame):
                self.data = [encode_pair(row.input, row.target) for row in df.itertuples()]

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, idx: int) -> Dict[str, List[int]]:
                return self.data[idx]

        train_dataset = PairDataset(train_df)
        test_dataset = PairDataset(test_df)

        return train_dataset, test_dataset

    def create_collator(self, tokenizer):
        """Create a collator for sequence-to-sequence batching with padding.

        This collator handles our specific present->past tense format:
        - Pads sequences to multiple of 8 for GPU efficiency
        - Uses -100 for label padding (PyTorch ignores these in loss)
        - Maintains input/attention_mask/labels format
        """
        class Seq2SeqPadCollator:
            def __init__(self, tok, mult=8):
                self.tok = tok
                self.mult = mult
            def __call__(self, feats):
                max_len = max(len(f["input_ids"]) for f in feats)
                if self.mult:
                    max_len = ((max_len + self.mult-1)//self.mult)*self.mult
                return {
                    "input_ids": torch.tensor([f["input_ids"] + [self.tok.pad_token_id]*(max_len-len(f["input_ids"])) for f in feats], dtype=torch.long),
                    "attention_mask": torch.tensor([f["attention_mask"]+[0]*(max_len-len(f["attention_mask"])) for f in feats], dtype=torch.long),
                    "labels": torch.tensor([f["labels"] + [-100]*(max_len-len(f["labels"])) for f in feats], dtype=torch.long),
                }

        return Seq2SeqPadCollator(tokenizer)

    def create_ablated_sentences(self, sentence_row: pd.Series) -> List[Dict[str, Any]]:
        """Create ablated versions of a sentence by substituting one word at a time in SVO order.
        
        Args:
            sentence_row: A pandas Series containing 'input', 'lang', 'plural', 'subj', 'obj', 'verb'
            
        Returns:
            List of dictionaries containing ablated sentences with metadata
        """

        if self.lexicon is None:
            self.load_lexicon()
            
        # Get source and target languages
        src_lang = sentence_row['lang']
        tgt_lang = 'nl' if src_lang == 'fr' else 'fr'
        
        # Parse the original sentence components
        plural = sentence_row['plural']
        det_type = "pl" if plural else "sgl"
        
        # Get determiners
        src_det = self.lexicon["DET"][src_lang][det_type]
        tgt_det = self.lexicon["DET"][tgt_lang][det_type]
        
        # Get nouns using parallel indices
        src_nouns = list(self.lexicon["NOUNS"][src_lang].keys())
        tgt_nouns = list(self.lexicon["NOUNS"][tgt_lang].keys())
        
        # Get subject translations
        subj_idx = src_nouns.index(sentence_row['subj'])
        tgt_subj_key = tgt_nouns[subj_idx]
        src_subj = self.lexicon["NOUNS"][src_lang][sentence_row['subj']][det_type]
        tgt_subj = self.lexicon["NOUNS"][tgt_lang][tgt_subj_key][det_type]
        
        # Get object translations
        obj_idx = src_nouns.index(sentence_row['obj'])
        tgt_obj_key = tgt_nouns[obj_idx]
        src_obj = self.lexicon["NOUNS"][src_lang][sentence_row['obj']][det_type]
        tgt_obj = self.lexicon["NOUNS"][tgt_lang][tgt_obj_key][det_type]
        
        # Get verb forms
        verb_key = sentence_row['verb']
        src_verbs = list(self.lexicon["VERBS"][src_lang].keys())
        tgt_verbs = list(self.lexicon["VERBS"][tgt_lang].keys())
        verb_idx = src_verbs.index(verb_key)
        tgt_verb_key = tgt_verbs[verb_idx]
        
        def get_verb_form(verb_info: dict, lang: str, is_plural: bool) -> str:
            if lang == "nl":
                return verb_info["present"]["zijpl" if is_plural else "hij"]
            else:  # fr
                return verb_info["present"]["ils" if is_plural else "il"]
        
        src_verb = get_verb_form(self.lexicon["VERBS"][src_lang][verb_key], src_lang, plural)
        tgt_verb = get_verb_form(self.lexicon["VERBS"][tgt_lang][tgt_verb_key], tgt_lang, plural)
        
        # Create ablated versions
        ablated = []
        
        # Original sentence
        base_sentence = f"{src_det} {src_subj} {src_verb} {src_det} {src_obj}"
        ablated.append({
            'input': base_sentence,
            'target': sentence_row['target'],
            'lang': src_lang,
            'plural': plural,
            'ablation': 'none',
            'subj': sentence_row['subj'],
            'obj': sentence_row['obj'],
            'verb': verb_key
        })
        
        # 1. Subject ablation (determinant + noun)
        subj_ablated = f"{tgt_det} {tgt_subj} {src_verb} {src_det} {src_obj}"
        ablated.append({
            'input': subj_ablated,
            'target': sentence_row['target'],
            'lang': src_lang,
            'plural': plural,
            'ablation': 'subject',
            'subj': sentence_row['subj'],
            'obj': sentence_row['obj'],
            'verb': verb_key
        })
        
        # 2. Verb ablation
        verb_ablated = f"{src_det} {src_subj} {tgt_verb} {src_det} {src_obj}"
        ablated.append({
            'input': verb_ablated,
            'target': sentence_row['target'],
            'lang': src_lang,
            'plural': plural,
            'ablation': 'verb',
            'subj': sentence_row['subj'],
            'obj': sentence_row['obj'],
            'verb': verb_key
        })
        
        # 3. Object ablation (determinant + noun)
        obj_ablated = f"{src_det} {src_subj} {src_verb} {tgt_det} {tgt_obj}"
        ablated.append({
            'input': obj_ablated,
            'target': sentence_row['target'],
            'lang': src_lang,
            'plural': plural,
            'ablation': 'object',
            'subj': sentence_row['subj'],
            'obj': sentence_row['obj'],
            'verb': verb_key
        })
        
        return ablated

    def create_ablated_dataset(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Create an ablated version of the dataset with one-word substitutions.
        
        Args:
            input_df: Input DataFrame containing sentences to ablate
            
        Returns:
            DataFrame containing original and ablated sentences
        """
        all_ablated = []
        for _, row in input_df.iterrows():
            ablated_sentences = self.create_ablated_sentences(row)
            all_ablated.extend(ablated_sentences)
        
        return pd.DataFrame(all_ablated)
