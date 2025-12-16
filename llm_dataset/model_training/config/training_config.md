# FLAN-T5 Fine-Tuning Configuration

## Model Selection

**Model:** google/flan-t5-base  
**Parameters:** 250M  
**Architecture:** Encoder-Decoder (Seq2Seq)

## LoRA Configuration

```yaml
lora_r: 8 # Rank of LoRA matrices
lora_alpha: 32 # Scaling factor
lora_dropout: 0.1 # Dropout for regularization
target_modules: ["q", "v"] # Apply to query and value attention layers
task_type: SEQ_2_SEQ_LM # Sequence-to-sequence task
```

## Training Hyperparameters

```yaml
learning_rate: 2e-5 # AdamW learning rate
batch_size: 4 # Per device batch size
num_epochs: 3 # Total training epochs
warmup_steps: 100 # Linear warmup steps
weight_decay: 0.01 # L2 regularization
max_input_length: 128 # Max tokens for input
max_output_length: 256 # Max tokens for output
```

## Optimization Settings

```yaml
optimizer: AdamW
lr_scheduler: linear
gradient_accumulation: 1
mixed_precision: fp16 # Use on GPU for speed
gradient_checkpointing: false
```

## Evaluation & Saving

```yaml
evaluation_strategy: epoch # Evaluate after each epoch
save_strategy: epoch # Save checkpoint after each epoch
save_total_limit: 2 # Keep only 2 best checkpoints
metric_for_best_model: rougeL
greater_is_better: true
load_best_model_at_end: true
```

## Metrics

```yaml
primary_metric: ROUGE-L # Main evaluation metric
additional_metrics:
  - ROUGE-1
  - ROUGE-2
  - Training Loss
  - Validation Loss
```

## Hardware Requirements

### Minimum (CPU Only)

- RAM: 16 GB
- Training Time: 20-30 hours
- **Not recommended for production**

### Recommended (GPU)

- GPU: NVIDIA T4 (16 GB) or better
- RAM: 16 GB
- Training Time: 2-4 hours
- Cost: Free on Google Colab

### Optimal (GPU with LoRA)

- GPU: NVIDIA V100 (16 GB) or A100 (40 GB)
- RAM: 32 GB
- Training Time: 1-2 hours
- Memory Usage: ~8-10 GB GPU RAM

## Dataset Configuration

```yaml
train_examples: 5289
val_examples: 661
test_examples: 662
total: 6612

avg_input_tokens: 31
avg_output_tokens: 38
max_sequence_length: 128 (input) + 256 (output)
```

## Expected Performance

### Training Metrics (per epoch)

```
Epoch 1:
  - Train Loss: ~2.5-3.0
  - Val Loss: ~2.0-2.5
  - ROUGE-L: ~0.35-0.45

Epoch 2:
  - Train Loss: ~1.5-2.0
  - Val Loss: ~1.5-2.0
  - ROUGE-L: ~0.45-0.55

Epoch 3:
  - Train Loss: ~1.0-1.5
  - Val Loss: ~1.3-1.8
  - ROUGE-L: ~0.50-0.65
```

### Final Test Set Performance (Target)

```
ROUGE-1: > 0.55
ROUGE-2: > 0.35
ROUGE-L: > 0.50
```

## Training Steps Calculation

```
Steps per epoch = train_examples / batch_size
                = 5289 / 4
                = 1323 steps

Total steps = 1323 × 3 epochs = 3969 steps
Logging every 100 steps = ~40 log entries per epoch
```

## Checkpoints

```yaml
checkpoint_format: pytorch_model.bin
save_location: model_training/checkpoints/
best_model_location: model_training/fine_tuned_model/
checkpoint_size: ~1 GB (with LoRA)
```

## Environment Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Check GPU

```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### Verify Model Access

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/flan-t5-base')"
```

## Running Training

### Basic Training

```bash
cd llm_dataset
python scripts/train_model.py
```

### Monitor Progress

```bash
# Watch logs in real-time
tail -f model_training/logs/*.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Resume from Checkpoint (if interrupted)

```python
# Modify train_model.py to include:
training_args = TrainingArguments(
    ...
    resume_from_checkpoint="model_training/checkpoints/checkpoint-XXXX"
)
```

## Output Files

After training completes:

```
model_training/
├── fine_tuned_model/
│   ├── pytorch_model.bin          # Model weights
│   ├── adapter_config.json        # LoRA config
│   ├── adapter_model.bin          # LoRA weights
│   ├── tokenizer_config.json      # Tokenizer config
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── test_results.json          # Test metrics
├── checkpoints/
│   ├── checkpoint-1323/           # Epoch 1
│   ├── checkpoint-2646/           # Epoch 2
│   └── checkpoint-3969/           # Epoch 3 (best)
└── logs/
    └── events.out.tfevents.*      # Training logs
```

## Troubleshooting

### Out of Memory (OOM)

```yaml
Solutions:
  - Reduce batch_size to 2 or 1
  - Reduce max_input_length to 96
  - Reduce max_output_length to 192
  - Enable gradient_checkpointing
  - Use FLAN-T5-small instead
```

### Slow Training

```yaml
Solutions:
  - Enable fp16 mixed precision
  - Increase batch_size (if memory allows)
  - Use gradient accumulation
  - Check GPU utilization (nvidia-smi)
  - Ensure using GPU not CPU
```

### Poor Convergence

```yaml
Solutions:
  - Increase num_epochs to 5
  - Adjust learning_rate (try 3e-5 or 1e-5)
  - Increase warmup_steps to 200
  - Check data quality
  - Verify tokenization
```

## Validation During Training

The model automatically evaluates on validation set after each epoch:

- Generates predictions for val set
- Computes ROUGE scores
- Saves checkpoint if best so far
- Logs metrics to logs directory

## Post-Training Steps

1. **Evaluate on Test Set**

   ```bash
   # Results automatically saved in test_results.json
   cat model_training/fine_tuned_model/test_results.json
   ```

2. **Test Inference**

   ```bash
   python scripts/claim_report_generation.py
   ```

3. **Generate Evaluation Report**

   ```bash
   python scripts/evaluate_model.py
   ```

4. **Deploy Model** (Optional)
   ```bash
   python scripts/app.py
   ```

## Notes

- Training is deterministic if you set random seed
- LoRA reduces trainable parameters by ~99%
- Model can be merged with base model later if needed
- Fine-tuned model will be ~1 GB in size
- First epoch is typically slowest (model initialization)

---

**Created:** November 27, 2025  
**Task:** Week 5 - Task 3: Fine-Tuning Setup  
**Next:** Task 4 - Model Evaluation
