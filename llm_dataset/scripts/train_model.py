"""
Week 5 - Task 3: Fine-Tuning Setup
Train FLAN-T5-base on claim report generation using LoRA for memory efficiency.

Requirements:
    pip install transformers datasets peft accelerate torch evaluate rouge-score
"""

import json
import os
from pathlib import Path
from datetime import datetime
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate
import numpy as np

# Configuration
class Config:
    # Model settings
    MODEL_NAME = "google/flan-t5-base"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "model_training"
    OUTPUT_DIR = DATA_DIR / "checkpoints"
    LOGS_DIR = DATA_DIR / "logs"
    FINAL_MODEL_DIR = DATA_DIR / "fine_tuned_model"
    
    # Training hyperparameters
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 1  # Reduced to prevent OOM
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4
    EVAL_BATCH_SIZE = 1  # Smaller eval batch
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    MAX_INPUT_LENGTH = 100  # Reduced from 128
    MAX_OUTPUT_LENGTH = 200  # Reduced from 256
    
    # LoRA configuration
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q", "v"]
    
    # Evaluation
    EVAL_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    SAVE_TOTAL_LIMIT = 2
    LOGGING_STEPS = 100
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_gpu_info():
    """Print GPU information."""
    print("\n" + "="*70)
    print("GPU INFORMATION")
    print("="*70)
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU available, training will use CPU (very slow)")
        print("   Recommendation: Use Google Colab or GPU-enabled environment")
    print("="*70 + "\n")

def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_datasets():
    """Load and prepare datasets."""
    print("📚 Loading datasets...")
    
    train_data = load_jsonl(Config.DATA_DIR / "clean_train.jsonl")
    val_data = load_jsonl(Config.DATA_DIR / "clean_val.jsonl")
    test_data = load_jsonl(Config.DATA_DIR / "clean_test.jsonl")
    
    print(f"   Training: {len(train_data)} examples")
    print(f"   Validation: {len(val_data)} examples")
    print(f"   Test: {len(test_data)} examples")
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    return train_dataset, val_dataset, test_dataset

def preprocess_function(examples, tokenizer):
    """Tokenize and preprocess examples."""
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input"],
        max_length=Config.MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    
    # Tokenize outputs (labels)
    labels = tokenizer(
        examples["output"],
        max_length=Config.MAX_OUTPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def load_model_and_tokenizer():
    """Load FLAN-T5 model and tokenizer."""
    print(f"\n🤖 Loading model: {Config.MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
    )
    
    print(f"   ✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer

def setup_lora(model):
    """Configure and apply LoRA to the model."""
    print("\n🔧 Configuring LoRA (Low-Rank Adaptation)...")
    
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   ✅ LoRA applied")
    print(f"   Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"   Total params: {total_params:,}")
    print(f"   Memory reduction: ~{(1 - trainable_params/total_params)*100:.0f}%")
    
    return model

def compute_metrics(eval_pred, tokenizer):
    """Compute ROUGE metrics for evaluation."""
    rouge = evaluate.load("rouge")
    
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }

def setup_training():
    """Set up training configuration."""
    print("\n⚙️  Setting up training configuration...")
    
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(Config.OUTPUT_DIR),
        eval_strategy=Config.EVAL_STRATEGY,
        save_strategy=Config.SAVE_STRATEGY,
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        logging_dir=str(Config.LOGS_DIR),
        logging_steps=Config.LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower loss is better
        warmup_steps=Config.WARMUP_STEPS,
        fp16=Config.DEVICE == "cuda",  # Use mixed precision on GPU
        report_to="none",  # Disable wandb/tensorboard for now
        push_to_hub=False,
        prediction_loss_only=True,  # Don't store predictions during eval (saves memory)
        eval_accumulation_steps=10,  # Process eval in smaller batches
        dataloader_pin_memory=False,  # Reduce memory usage
    )
    
    print(f"   Output directory: {Config.OUTPUT_DIR}")
    print(f"   Logging directory: {Config.LOGS_DIR}")
    print(f"   Batch size: {Config.BATCH_SIZE} (per device)")
    print(f"   Gradient accumulation: {Config.GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"   Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Eval batch size: {Config.EVAL_BATCH_SIZE}")
    print(f"   Epochs: {Config.NUM_EPOCHS}")
    print(f"   Learning rate: {Config.LEARNING_RATE}")
    print(f"   Mixed precision (fp16): {training_args.fp16}")
    print(f"   Max input length: {Config.MAX_INPUT_LENGTH}")
    print(f"   Max output length: {Config.MAX_OUTPUT_LENGTH}")
    
    return training_args

def train_model():
    """Main training function."""
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("WEEK 5 - TASK 3: FINE-TUNING FLAN-T5 FOR CLAIM GENERATION")
    print("="*70)
    
    # GPU info
    print_gpu_info()
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Apply LoRA
    model = setup_lora(model)
    
    # Preprocess datasets
    print("\n📝 Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    print("   ✅ Tokenization complete")
    
    # Setup training
    training_args = setup_training()
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Create trainer
    print("\n🎯 Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # Note: compute_metrics not used with prediction_loss_only=True
        # Full evaluation will run after training via evaluate_model.py
    )
    
    print("   ✅ Trainer initialized")
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        print("\n🧹 Clearing GPU cache...")
        torch.cuda.empty_cache()
        print(f"   GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Train
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated duration: 2-4 hours (depends on GPU)")
    print("="*70 + "\n")
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ CUDA out of memory error!")
            print("💡 Recommendations:")
            print("   1. Restart runtime: Runtime → Restart runtime")
            print("   2. Reduce batch size further in train_model.py")
            print("   3. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
            raise
        else:
            raise
    
    # Save final model
    print("\n💾 Saving fine-tuned model...")
    Config.FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    trainer.model.save_pretrained(Config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(Config.FINAL_MODEL_DIR)
    
    print(f"   ✅ Model saved to: {Config.FINAL_MODEL_DIR}")
    
    # Final evaluation on test set
    print("\n📊 Evaluating on test set...")
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names,
    )
    
    test_results = trainer.evaluate(test_dataset)
    
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    for key, value in test_results.items():
        print(f"   {key}: {value:.4f}")
    print("="*70)
    
    # Save test results
    results_file = Config.FINAL_MODEL_DIR / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Training summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE ✅")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Model saved: {Config.FINAL_MODEL_DIR}")
    print("="*70 + "\n")
    
    return trainer, test_results

if __name__ == "__main__":
    try:
        trainer, results = train_model()
        print("✨ Fine-tuning completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
