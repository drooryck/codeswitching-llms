import pandas as pd
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from typing import Dict, List, Any
import logging
from transformers import logging as transformers_logging
import asyncio
from tqdm import tqdm

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

async def process_batch(model, tokenizer, batch_rows, device) -> List[Dict[str, Any]]:
    predictions = []
    
    # Process all prompts in the batch at once
    prompts = [f"<sos> {row.input} <sep>" for row in batch_rows]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            num_beams=4,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Process predictions
    for i, row in enumerate(batch_rows):
        pred = tokenizer.decode(outputs[i], skip_special_tokens=False)
        pred = pred.split("<sep>")[1].replace("<eos>", "").strip()
        
        predictions.append({
            'language': row.lang,
            'input': row.input,
            'gold': row.target,
            'prediction': pred,
            'ablation': row.ablation
        })
    
    return predictions

async def run_inference_on_ablated(model_path: str, test_df: pd.DataFrame) -> List[Dict[str, Any]]:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Process in batches
    batch_size = 32
    all_predictions = []
    tasks = []
    
    # Create batches
    for i in tqdm(range(0, len(test_df), batch_size)):
        batch_rows = [row for row in test_df.iloc[i:i+batch_size].itertuples()]
        task = asyncio.create_task(process_batch(model, tokenizer, batch_rows, device))
        tasks.append(task)
    
    # Wait for all batches to complete
    batch_predictions = await asyncio.gather(*tasks)
    
    # Flatten predictions
    for batch_pred in batch_predictions:
        all_predictions.extend(batch_pred)
    
    return all_predictions

if __name__ == "__main__":
    # Load ablated data
    abl = pd.read_csv("/n/home06/drooryck/codeswitching-llms/july_aug_exp/scripts/data/ablated_test.csv")
    
    # Run inference
    model_path = "/n/home06/drooryck/codeswitching-llms/july_aug_exp/results/sep12.3/p50_run1/final"
    predictions = asyncio.run(run_inference_on_ablated(model_path, abl))
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv("/n/home06/drooryck/codeswitching-llms/july_aug_exp/results/sep12.3/p50_run1/ablation_predictions_fromslurm_p.csv", index=False)