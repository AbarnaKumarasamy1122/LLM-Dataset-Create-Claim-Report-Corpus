# Quick Start Guide - Week 5 Tasks

## ✅ COMPLETED: Task 1 - Data Preparation

**Status:** 100% Complete  
**Output:** 6,612 clean training examples ready for fine-tuning  
**Location:** `model_training/clean_*.jsonl`

---

## 🚀 NEXT: Task 2 - Choose a Base Model

### Recommended Models (in order of GPU requirements):

#### 1. FLAN-T5-small (Recommended for Start)

- **Size:** 80M parameters
- **GPU RAM:** 4-6 GB
- **Training Time:** 30-60 min (3 epochs)
- **Best For:** Quick prototyping, limited GPU
- **Model ID:** `google/flan-t5-small`

#### 2. FLAN-T5-base (Recommended for MVP) ⭐

- **Size:** 250M parameters
- **GPU RAM:** 12-16 GB (8-10 GB with LoRA)
- **Training Time:** 2-4 hours (3 epochs)
- **Best For:** Production-ready quality, balanced resources
- **Model ID:** `google/flan-t5-base`

#### 3. FLAN-T5-large

- **Size:** 780M parameters
- **GPU RAM:** 24+ GB (16 GB with LoRA)
- **Training Time:** 6-10 hours (3 epochs)
- **Best For:** Higher quality outputs
- **Model ID:** `google/flan-t5-large`

#### 4. Mistral-7B

- **Size:** 7B parameters
- **GPU RAM:** 32+ GB (16-20 GB with LoRA)
- **Training Time:** 8-12 hours (3 epochs)
- **Best For:** State-of-art quality, ample resources
- **Model ID:** `mistralai/Mistral-7B-v0.1`

#### 5. Llama-3-8B

- **Size:** 8B parameters
- **GPU RAM:** 40+ GB (20-24 GB with LoRA)
- **Training Time:** 10-15 hours (3 epochs)
- **Best For:** Best quality, research use
- **Model ID:** `meta-llama/Meta-Llama-3-8B`

### Decision Matrix

| Factor     | FLAN-T5-small | FLAN-T5-base | FLAN-T5-large | Mistral-7B | Llama-3-8B |
| ---------- | ------------- | ------------ | ------------- | ---------- | ---------- |
| Quality    | ⭐⭐⭐        | ⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed      | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐     | ⭐⭐⭐        | ⭐⭐       | ⭐⭐       |
| GPU Need   | ⭐            | ⭐⭐⭐       | ⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Easy Setup | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐      | ⭐⭐⭐     | ⭐⭐⭐     |

### To Complete Task 2:

1. **Create model choice file:**

```bash
# Navigate to config folder
cd model_training/config

# Create model_choice.txt
echo "Model: google/flan-t5-base" > model_choice.txt
echo "Reason: Balanced performance and resource efficiency for MVP" >> model_choice.txt
```

2. **Or use our recommended selection (FLAN-T5-base):**
   - Good quality outputs for claim reports
   - Reasonable training time (2-4 hours)
   - Compatible with LoRA for memory efficiency
   - Well-documented and stable

---

## 📋 Task 3 Preview: Fine-Tuning Setup

### Required Packages:

```bash
pip install transformers
pip install datasets
pip install peft
pip install accelerate
pip install torch
pip install bitsandbytes  # For quantization (optional)
```

### Training Configuration:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./model_training/checkpoints",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./model_training/logs",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

### LoRA Configuration (Memory Efficient):

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=32,          # Scaling factor
    target_modules=["q", "v"],  # Which layers to adapt
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
```

---

## 🎯 Current Progress Tracker

### Week 5 Tasks:

- [x] **Task 1:** Data Preparation ✅ (100% complete)
- [ ] **Task 2:** Choose a Base Model (Next)
- [ ] **Task 3:** Fine-Tuning Setup
- [ ] **Task 4:** Model Evaluation
- [ ] **Task 5:** Inference Pipeline
- [ ] **Task 6:** Deployment Prep (Optional)

### Expected Timeline:

- **Task 2:** 30 minutes (model selection + documentation)
- **Task 3:** 1-2 hours (setup + initial training)
- **Task 4:** 1 hour (evaluation metrics)
- **Task 5:** 1-2 hours (inference script)
- **Task 6:** 2-4 hours (API + deployment)

**Total Estimated Time:** 1-2 days for core tasks (Tasks 2-5)

---

## 💡 Tips for Success

### GPU Considerations:

- **Free Options:** Google Colab (T4, 15GB), Kaggle (P100, 16GB)
- **Paid Options:** Google Colab Pro (A100, 40GB), AWS SageMaker, RunPod
- **Local:** Check `nvidia-smi` for available GPU memory

### Memory Optimization:

1. Use LoRA/PEFT (reduces memory by 50-70%)
2. Enable gradient checkpointing
3. Use mixed precision (fp16/bf16)
4. Reduce batch size if OOM errors occur

### Training Best Practices:

1. Start with small model (FLAN-T5-small) to test pipeline
2. Monitor validation loss to prevent overfitting
3. Save checkpoints regularly
4. Log metrics to track progress
5. Test inference after each epoch

---

## 📚 Resources

### Documentation:

- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PEFT/LoRA: https://huggingface.co/docs/peft
- FLAN-T5 Paper: https://arxiv.org/abs/2210.11416

### Example Notebooks:

- FLAN-T5 Fine-tuning: https://huggingface.co/docs/transformers/model_doc/flan-t5
- LoRA Tutorial: https://huggingface.co/docs/peft/task_guides/seq2seq-prefix-tuning

### Our Scripts:

- Data Preparation: `scripts/prepare_data_for_finetuning.py`
- Validation: `scripts/validate_and_tokenize.py`
- Dry Run (Week 4): `scripts/fine_tune_dry_run.py`

---

## ❓ Quick Commands

```bash
# Check GPU availability
nvidia-smi

# Install required packages
pip install -r requirements.txt

# Test model loading (after Task 2)
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
           model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base'); \
           print('Model loaded successfully!')"

# Run training (after Task 3)
python scripts/train_model.py --config model_training/config/training_config.yaml

# Run evaluation (after Task 4)
python scripts/evaluate_model.py --checkpoint model_training/checkpoints/best_model

# Test inference (after Task 5)
python scripts/generate_claim.py --input "shipment_id=SHP-12345; damage_type=dent; severity=high"
```

---

## ✅ Checklist Before Starting Task 2

- [x] Task 1 completed and validated
- [x] Clean datasets available in `model_training/`
- [x] Folder structure ready
- [ ] GPU access confirmed (check with `nvidia-smi`)
- [ ] Python environment ready (Python 3.8+)
- [ ] Required packages listed in requirements.txt
- [ ] Reviewed model options above
- [ ] Selected target model based on GPU capacity

---

**Ready to proceed with Task 2!** 🚀

_Last Updated: November 27, 2025_
