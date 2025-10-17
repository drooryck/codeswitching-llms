#!/usr/bin/env python
"""
Script to generate training and test data for language experiments.
"""
import sys
from pathlib import Path

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.dataset_manager import DatasetManager


def main():
    """Generate data using DatasetManager."""
    data_dir = root_dir / "data"
    data_manager = DatasetManager(data_dir)

    # Load lexicon
    lexicon = data_manager.load_lexicon()
    print(f"Loaded lexicon with {len(lexicon)} entries")

    # Generate and save data
    train_df, test_df = data_manager.make_and_save_testing_and_training_data()
    print(f"Generated {len(train_df)} training samples and {len(test_df)} test samples")


if __name__ == "__main__":
    main()
