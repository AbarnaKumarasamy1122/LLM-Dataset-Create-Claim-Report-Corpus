# Dry Run Report — LLM Claim Report Dataset

**Dataset:** MVTec + Synthetic Metadata (Week 4)
**Model:** distilgpt2 (1 epoch)
**Records used:** 5289 train, 661 val, 662 test

## Observations
✅ Model learns structure: “Package [ID] shows [severity] [damage_type]…”
✅ Average BLEU score: 0.72  
⚠️ Occasionally repeats sentences for high severity cases.  
⚠️ Slight inconsistency in liability phrasing.

## Next Steps
- Add more template diversity.
- Increase training epochs to 3.
- Try larger model (Llama-2 / Mistral-7B) for production.
