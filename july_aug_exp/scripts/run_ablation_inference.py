# Save this as /n/home06/drooryck/codeswitching-llms/july_aug_exp/scripts/run_ablation_inference.py
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from typing import Dict, List, Any
import logging
from transformers import logging as transformers_logging
import sys
from pathlib import Path

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_inference_on_ablated(model_path: str, test_df: pd.DataFrame) -> List[Dict[str, Any]]:
    print(f"Loading model from {model_path}")
    
    # Check if model files exist
    config_path = Path(model_path) / "config.json"
    model_file = Path(model_path) / "model.safetensors"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}")
        
    print("Found model files, loading...")
    
    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    total = len(test_df)
    
    print(f"Starting inference on {total} sentences...")
    for i, row in enumerate(test_df.itertuples(), 1):
        if i % 100 == 0:
            print(f"Processing sentence {i}/{total}")
            
        prompt = f"<sos> {row.input} <sep>"
        
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                num_beams=4,
                eos_token_id=tokenizer.eos_token_id
            )
            
        pred = tokenizer.decode(outputs[0], skip_special_tokens=False)
        pred = pred.split("<sep>")[1].replace("<eos>", "").strip()
        
        predictions.append({
            'language': row.lang,
            'input': row.input,
            'gold': row.target,
            'prediction': pred,
            'ablation': row.ablation
        })
    
    return predictions

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_ablation_inference.py <run_directory>")
        sys.exit(1)
        
    run_dir = Path(sys.argv[1])
    print(f"Processing run directory: {run_dir}")
    
    # Model path is the 'final' subdirectory of the run directory
    model_path = run_dir / "final"
    
    # Load ablated data
    print("Loading ablated test data...")
    abl = pd.read_csv("/n/home06/drooryck/codeswitching-llms/july_aug_exp/scripts/data/ablated_test.csv")
    
    # Run inference
    predictions = run_inference_on_ablated(str(model_path), abl)
    
    # Save predictions in the run directory
    output_path = run_dir / "ablation_predictions.csv"
    print(f"Saving predictions to {output_path}")
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_path, index=False)
    
    print("Done!")