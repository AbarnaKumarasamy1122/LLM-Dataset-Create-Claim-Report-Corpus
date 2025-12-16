"""
Week 5 - Task 4: Model Evaluation
Comprehensive evaluation of fine-tuned FLAN-T5 model for claim report generation.

Evaluates on:
- BLEU scores (unigram, bigram, trigram, 4-gram)
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Custom metrics (accuracy, relevance, readability, consistency)
- Sample analysis

Output: evaluation_report.csv
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from evaluate import load
from tqdm import tqdm

# Configuration
class Config:
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / "model_training" / "fine_tuned_model"
    DATA_DIR = BASE_DIR / "model_training"
    OUTPUT_DIR = BASE_DIR / "model_training"
    
    TEST_FILE = DATA_DIR / "clean_test.jsonl"
    EVAL_REPORT = OUTPUT_DIR / "evaluation_report.csv"
    SAMPLES_FILE = OUTPUT_DIR / "evaluation_samples.txt"
    
    # Generation parameters
    MAX_INPUT_LENGTH = 128
    MAX_OUTPUT_LENGTH = 256
    NUM_BEAMS = 4
    TEMPERATURE = 0.7
    
    # Evaluation settings
    NUM_SAMPLES_TO_SHOW = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_test_data():
    """Load test dataset."""
    print("📊 Loading test data...")
    data = []
    with open(Config.TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"   ✅ Loaded {len(data)} test examples")
    return data

def load_model_and_tokenizer():
    """Load fine-tuned model and tokenizer."""
    print(f"\n🤖 Loading fine-tuned model from {Config.MODEL_DIR}")
    
    if not Config.MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at {Config.MODEL_DIR}. "
            "Please train the model first using: python scripts/train_model.py"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        Config.MODEL_DIR,
        torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
    )
    model.to(Config.DEVICE)
    model.eval()
    
    print(f"   ✅ Model loaded on {Config.DEVICE}")
    return model, tokenizer

def generate_predictions(model, tokenizer, test_data):
    """Generate predictions for all test examples."""
    print("\n🔮 Generating predictions...")
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for example in tqdm(test_data, desc="Generating"):
            # Tokenize input
            inputs = tokenizer(
                example['input'],
                max_length=Config.MAX_INPUT_LENGTH,
                truncation=True,
                return_tensors="pt"
            ).to(Config.DEVICE)
            
            # Generate output
            outputs = model.generate(
                **inputs,
                max_length=Config.MAX_OUTPUT_LENGTH,
                num_beams=Config.NUM_BEAMS,
                temperature=Config.TEMPERATURE,
                do_sample=False,
                early_stopping=True,
            )
            
            # Decode prediction
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(example['output'])
    
    print(f"   ✅ Generated {len(predictions)} predictions")
    return predictions, references

def compute_bleu_scores(predictions, references):
    """Compute BLEU scores."""
    print("\n📈 Computing BLEU scores...")
    
    bleu = load("bleu")
    
    # BLEU-1, BLEU-2, BLEU-3, BLEU-4
    bleu_scores = {}
    for n in [1, 2, 3, 4]:
        result = bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references],
            max_order=n
        )
        bleu_scores[f'bleu_{n}'] = result['bleu']
        print(f"   BLEU-{n}: {result['bleu']:.4f}")
    
    return bleu_scores

def compute_rouge_scores(predictions, references):
    """Compute ROUGE scores."""
    print("\n📈 Computing ROUGE scores...")
    
    rouge = load("rouge")
    
    result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    
    rouge_scores = {
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL'],
        'rougeLsum': result['rougeLsum'],
    }
    
    for key, value in rouge_scores.items():
        print(f"   {key.upper()}: {value:.4f}")
    
    return rouge_scores

def compute_custom_metrics(predictions, references, test_data):
    """Compute custom evaluation metrics."""
    print("\n📈 Computing custom metrics...")
    
    metrics = {
        'avg_pred_length': np.mean([len(p.split()) for p in predictions]),
        'avg_ref_length': np.mean([len(r.split()) for r in references]),
        'exact_matches': sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()),
        'empty_predictions': sum(1 for p in predictions if not p.strip()),
    }
    
    # Check metadata consistency
    consistent_count = 0
    for pred, example in zip(predictions, test_data):
        pred_lower = pred.lower()
        input_text = example['input']
        
        # Extract key fields from input
        fields_to_check = []
        for field in ['shipment_id', 'damage_type', 'severity']:
            if f'{field}=' in input_text:
                value = input_text.split(f'{field}=')[1].split(';')[0].strip()
                fields_to_check.append(value.lower())
        
        # Check if key fields appear in prediction
        if all(field in pred_lower for field in fields_to_check):
            consistent_count += 1
    
    metrics['metadata_consistency'] = consistent_count / len(predictions)
    
    # Readability: average sentence length
    sentence_lengths = []
    for pred in predictions:
        sentences = pred.split('.')
        for sent in sentences:
            words = sent.strip().split()
            if words:
                sentence_lengths.append(len(words))
    
    metrics['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
    
    print(f"   Avg prediction length: {metrics['avg_pred_length']:.1f} words")
    print(f"   Avg reference length: {metrics['avg_ref_length']:.1f} words")
    print(f"   Exact matches: {metrics['exact_matches']}")
    print(f"   Empty predictions: {metrics['empty_predictions']}")
    print(f"   Metadata consistency: {metrics['metadata_consistency']:.2%}")
    print(f"   Avg sentence length: {metrics['avg_sentence_length']:.1f} words")
    
    return metrics

def analyze_by_damage_type(predictions, references, test_data):
    """Analyze performance by damage type."""
    print("\n📈 Analyzing by damage type...")
    
    rouge = load("rouge")
    by_damage = defaultdict(lambda: {'count': 0, 'rouge1': [], 'rouge2': [], 'rougeL': []})
    
    for pred, ref, example in zip(predictions, references, test_data):
        input_text = example['input']
        
        # Extract damage type
        if 'damage_type=' in input_text:
            damage_type = input_text.split('damage_type=')[1].split(';')[0].strip()
            
            # Compute ROUGE for this example
            result = rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)
            
            by_damage[damage_type]['count'] += 1
            by_damage[damage_type]['rouge1'].append(result['rouge1'])
            by_damage[damage_type]['rouge2'].append(result['rouge2'])
            by_damage[damage_type]['rougeL'].append(result['rougeL'])
    
    # Compute averages
    damage_analysis = {}
    for damage_type, scores in by_damage.items():
        damage_analysis[damage_type] = {
            'count': scores['count'],
            'rouge1': np.mean(scores['rouge1']),
            'rouge2': np.mean(scores['rouge2']),
            'rougeL': np.mean(scores['rougeL']),
        }
        print(f"   {damage_type}: n={scores['count']}, ROUGE-L={np.mean(scores['rougeL']):.4f}")
    
    return damage_analysis

def save_evaluation_report(all_scores, damage_analysis, test_data_count):
    """Save comprehensive evaluation report to CSV."""
    print(f"\n💾 Saving evaluation report to {Config.EVAL_REPORT}")
    
    # Prepare rows for CSV
    rows = []
    
    # Overall metrics
    rows.append({
        'Metric': 'Test Examples',
        'Value': test_data_count,
        'Category': 'Dataset'
    })
    
    # BLEU scores
    for key, value in all_scores.get('bleu', {}).items():
        rows.append({
            'Metric': key.upper(),
            'Value': f"{value:.4f}",
            'Category': 'BLEU'
        })
    
    # ROUGE scores
    for key, value in all_scores.get('rouge', {}).items():
        rows.append({
            'Metric': key.upper(),
            'Value': f"{value:.4f}",
            'Category': 'ROUGE'
        })
    
    # Custom metrics
    for key, value in all_scores.get('custom', {}).items():
        if isinstance(value, float):
            rows.append({
                'Metric': key.replace('_', ' ').title(),
                'Value': f"{value:.4f}" if value < 1 else f"{value:.1f}",
                'Category': 'Custom'
            })
        else:
            rows.append({
                'Metric': key.replace('_', ' ').title(),
                'Value': str(value),
                'Category': 'Custom'
            })
    
    # Damage type breakdown
    for damage_type, scores in damage_analysis.items():
        rows.append({
            'Metric': f"{damage_type} (n={scores['count']})",
            'Value': f"R1={scores['rouge1']:.3f}, R2={scores['rouge2']:.3f}, RL={scores['rougeL']:.3f}",
            'Category': 'By Damage Type'
        })
    
    # Write CSV
    with open(Config.EVAL_REPORT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Metric', 'Value', 'Category'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"   ✅ Report saved: {Config.EVAL_REPORT}")

def save_sample_predictions(predictions, references, test_data):
    """Save sample predictions for manual review."""
    print(f"\n💾 Saving sample predictions to {Config.SAMPLES_FILE}")
    
    num_samples = min(Config.NUM_SAMPLES_TO_SHOW, len(predictions))
    
    with open(Config.SAMPLES_FILE, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION SAMPLES - Manual Review\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Examples: {len(predictions)}\n")
        f.write(f"Showing: {num_samples} samples\n\n")
        
        for i in range(num_samples):
            f.write("="*80 + "\n")
            f.write(f"SAMPLE {i+1}/{num_samples}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"INPUT:\n{test_data[i]['input']}\n\n")
            f.write(f"REFERENCE (Ground Truth):\n{references[i]}\n\n")
            f.write(f"PREDICTION (Model Output):\n{predictions[i]}\n\n")
            
            # Compute ROUGE for this example
            rouge = load("rouge")
            result = rouge.compute(predictions=[predictions[i]], references=[references[i]], use_stemmer=True)
            
            f.write(f"ROUGE Scores:\n")
            f.write(f"  ROUGE-1: {result['rouge1']:.4f}\n")
            f.write(f"  ROUGE-2: {result['rouge2']:.4f}\n")
            f.write(f"  ROUGE-L: {result['rougeL']:.4f}\n\n")
    
    print(f"   ✅ Samples saved: {Config.SAMPLES_FILE}")

def print_summary(all_scores):
    """Print evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    # BLEU scores
    print("\n📊 BLEU Scores:")
    for key, value in all_scores['bleu'].items():
        print(f"   {key.upper()}: {value:.4f}")
    
    # ROUGE scores
    print("\n📊 ROUGE Scores:")
    for key, value in all_scores['rouge'].items():
        print(f"   {key.upper()}: {value:.4f}")
    
    # Custom metrics
    print("\n📊 Custom Metrics:")
    for key, value in all_scores['custom'].items():
        if isinstance(value, float):
            if value < 1:
                print(f"   {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Quality assessment
    rouge_l = all_scores['rouge']['rougeL']
    print("\n🎯 Quality Assessment:")
    if rouge_l >= 0.60:
        print("   ✅ EXCELLENT - Human-like quality")
    elif rouge_l >= 0.50:
        print("   ✅ GOOD - Coherent and accurate reports")
    elif rouge_l >= 0.40:
        print("   ⚠️  ACCEPTABLE - Captures main information")
    else:
        print("   ❌ NEEDS IMPROVEMENT - Consider retraining")
    
    print("\n" + "="*70)

def main():
    """Main evaluation function."""
    print("\n" + "="*70)
    print("WEEK 5 - TASK 4: MODEL EVALUATION")
    print("="*70)
    
    start_time = datetime.now()
    
    try:
        # Load data
        test_data = load_test_data()
        
        # Load model
        model, tokenizer = load_model_and_tokenizer()
        
        # Generate predictions
        predictions, references = generate_predictions(model, tokenizer, test_data)
        
        # Compute BLEU scores
        bleu_scores = compute_bleu_scores(predictions, references)
        
        # Compute ROUGE scores
        rouge_scores = compute_rouge_scores(predictions, references)
        
        # Compute custom metrics
        custom_metrics = compute_custom_metrics(predictions, references, test_data)
        
        # Analyze by damage type
        damage_analysis = analyze_by_damage_type(predictions, references, test_data)
        
        # Compile all scores
        all_scores = {
            'bleu': bleu_scores,
            'rouge': rouge_scores,
            'custom': custom_metrics,
        }
        
        # Save reports
        save_evaluation_report(all_scores, damage_analysis, len(test_data))
        save_sample_predictions(predictions, references, test_data)
        
        # Print summary
        print_summary(all_scores)
        
        # Completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱️  Evaluation time: {duration}")
        print(f"\n✅ Evaluation complete!")
        print(f"   Report: {Config.EVAL_REPORT}")
        print(f"   Samples: {Config.SAMPLES_FILE}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please train the model first:")
        print("   python scripts/train_model.py")
        return 1
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
