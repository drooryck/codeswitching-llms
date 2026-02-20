"""
Stripped-down data preparation function for debugging and inspection.
Can be run in a Jupyter notebook to examine data processing logic.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler
from typing import Tuple
from .dataset_manager import DatasetManager
from .model_config import ModelConfig


def prepare_data_debug(
    prop: float,
    run_id: int,
    eval_prop: float,
    data_manager: DatasetManager,
    config: ModelConfig,
    batch_size: int = 32
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DataLoader]:
    """
    Prepare training data following run_single_super_debug logic (without wandb/timing).
    
    Args:
        prop: French proportion (0.0 to 1.0)
        run_id: Random seed for reproducibility
        eval_prop: Validation split proportion
        data_manager: DatasetManager instance
        config: ModelConfig instance
        batch_size: Batch size for DataLoader
        
    Returns:
        Tuple of (train_df, val_df, test_df, train_dataloader)
        - train_df: Final training dataframe (shuffled)
        - val_df: Validation dataframe
        - test_df: Test dataframe
        - train_dataloader: DataLoader with SequentialSampler
    """
    # Set seed
    np.random.seed(run_id)
    
    # Load data
    train_df_full = pd.read_csv(data_manager.data_dir / "train.csv")
    test_df_full = pd.read_csv(data_manager.data_dir / "test.csv")
    
    print(f"Initial train set: {len(train_df_full)} total | "
          f"FR: {(train_df_full['lang'] == 'fr').sum()} | "
          f"NL: {(train_df_full['lang'] == 'nl').sum()}")
    
    rng = np.random.RandomState(run_id)
    
    # Balance initial dataset FIRST - keep min(FR_count, NL_count) from each language
    fr_count = (train_df_full['lang'] == 'fr').sum()
    nl_count = (train_df_full['lang'] == 'nl').sum()
    min_count = min(fr_count, nl_count)
    fr_df = train_df_full[train_df_full['lang'] == 'fr'].sample(n=min_count, random_state=run_id)
    nl_df = train_df_full[train_df_full['lang'] == 'nl'].sample(n=min_count, random_state=run_id)
    train_df_full = pd.concat([fr_df, nl_df], ignore_index=True).reset_index(drop=True)
    
    print(f"Balanced train set: {len(train_df_full)} total | "
          f"FR: {(train_df_full['lang'] == 'fr').sum()} | "
          f"NL: {(train_df_full['lang'] == 'nl').sum()}")
    
    # Stratified validation split (on balanced data)
    train_df_full, val_df = train_test_split(
        train_df_full,
        test_size=eval_prop,
        random_state=run_id,
        shuffle=True,
        stratify=train_df_full['lang']
    )
    train_df_full = train_df_full.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f"Validation set: {len(val_df)} total | "
          f"FR: {(val_df['lang'] == 'fr').sum()} | "
          f"NL: {(val_df['lang'] == 'nl').sum()}")
    print(f"Train pool (post-val): {len(train_df_full)} total | "
          f"FR: {(train_df_full['lang'] == 'fr').sum()} | "
          f"NL: {(train_df_full['lang'] == 'nl').sum()}")
    
    # Apply global_key to training data only (after validation split)
    train_df_full["global_key"] = rng.random(len(train_df_full))
    train_df_full["orig_idx"] = np.arange(len(train_df_full))
    
    train_df_full["lang_rank"] = (
        train_df_full.groupby("lang")["global_key"]
                    .rank(method="first")
                    .astype(int)
    )
    
    total_budget = min(
        (train_df_full.lang == "fr").sum(),
        (train_df_full.lang == "nl").sum()
    )
    
    want_fr = int(total_budget * prop)
    want_nl = total_budget - want_fr
    
    fr_take = train_df_full[
        (train_df_full.lang == "fr") & (train_df_full.lang_rank <= want_fr)
    ]
    nl_take = train_df_full[
        (train_df_full.lang == "nl") & (train_df_full.lang_rank <= want_nl)
    ]
    
    train_df = (
        pd.concat([fr_take, nl_take], ignore_index=True)
        .sample(frac=1, random_state=run_id)
        .reset_index(drop=True)
    )
    
    print(f"\nFinal training set:")
    print(f"  Total: {len(train_df)} | "
          f"FR: {len(fr_take)} ({len(fr_take)/len(train_df)*100:.1f}%) | "
          f"NL: {len(nl_take)} ({len(nl_take)/len(train_df)*100:.1f}%)")
    print(f"  Target proportion: FR={prop*100:.1f}%, NL={100-prop*100:.1f}%")
    
    # Create tokenizer and datasets
    tokenizer = data_manager.build_tokenizer()
    train_dataset, val_dataset = data_manager.create_pytorch_datasets(
        train_df, val_df, tokenizer
    )
    collator = data_manager.create_collator(tokenizer)
    
    # Create DataLoader with SequentialSampler (preserves shuffled order)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(train_dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"\nCreated DataLoader with {len(train_dataloader)} batches")
    print(f"First 20 training examples (showing language interlacing):")
    for i in range(min(20, len(train_df))):
        lang = train_df.iloc[i]['lang']
        input_text = train_df.iloc[i]['input'][:50]  # First 50 chars
        print(f"  [{i:3d}] {lang.upper():2s}: {input_text}...")
    
    return train_df, val_df, test_df_full, train_dataloader
