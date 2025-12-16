"""
Week 5 - Task 5: Inference Pipeline
Generate claim reports from shipment metadata using fine-tuned model.

Usage:
    # Single prediction
    python scripts/claim_report_generation.py --input "shipment_id=SHP-123; damage_type=dent; severity=high"
    
    # From JSON file
    python scripts/claim_report_generation.py --input_file input.json
    
    # Batch processing
    python scripts/claim_report_generation.py --batch batch_inputs.jsonl --output batch_outputs.jsonl
    
    # Interactive mode
    python scripts/claim_report_generation.py --interactive
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class ClaimReportGenerator:
    """Claim report generation using fine-tuned FLAN-T5 model."""
    
    def __init__(self, model_path=None):
        """
        Initialize the generator.
        
        Args:
            model_path: Path to fine-tuned model. If None, uses default location.
        """
        if model_path is None:
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / "model_training" / "fine_tuned_model"
        
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🤖 Loading model from {self.model_path}")
        print(f"   Device: {self.device}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first: python scripts/train_model.py"
            )
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        
        print("   ✅ Model loaded successfully")
    
    def format_input(self, metadata_dict):
        """
        Format metadata dictionary into model input string.
        
        Args:
            metadata_dict: Dictionary with shipment metadata
            
        Returns:
            Formatted input string
        """
        # Required fields
        required_fields = ['shipment_id', 'damage_type', 'severity']
        for field in required_fields:
            if field not in metadata_dict:
                raise ValueError(f"Missing required field: {field}")
        
        # Optional fields
        optional_fields = ['image_id', 'vendor', 'shipment_stage']
        
        # Build input string
        parts = []
        for field in required_fields + optional_fields:
            if field in metadata_dict:
                parts.append(f"{field}={metadata_dict[field]}")
        
        return "; ".join(parts)
    
    def generate_claim(self, input_text, max_length=256, num_beams=4, temperature=0.7):
        """
        Generate a claim report from input text.
        
        Args:
            input_text: Formatted input string with metadata
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            Generated claim report text
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=False,
                early_stopping=True,
            )
        
        # Decode and return
        claim_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return claim_text
    
    def generate_from_dict(self, metadata_dict, **kwargs):
        """
        Generate claim report from metadata dictionary.
        
        Args:
            metadata_dict: Dictionary with shipment metadata
            **kwargs: Additional generation parameters
            
        Returns:
            Generated claim report text
        """
        input_text = self.format_input(metadata_dict)
        return self.generate_claim(input_text, **kwargs)
    
    def batch_generate(self, input_list, **kwargs):
        """
        Generate claim reports for multiple inputs.
        
        Args:
            input_list: List of input strings or metadata dictionaries
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated claim reports
        """
        results = []
        for item in input_list:
            if isinstance(item, dict):
                text = self.generate_from_dict(item, **kwargs)
            else:
                text = self.generate_claim(item, **kwargs)
            results.append(text)
        return results

def load_json_input(filepath):
    """Load input from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl_batch(filepath):
    """Load batch inputs from JSONL file."""
    inputs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))
    return inputs

def save_jsonl_batch(outputs, metadata_list, filepath):
    """Save batch outputs to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for metadata, output in zip(metadata_list, outputs):
            entry = {
                'input': metadata if isinstance(metadata, str) else json.dumps(metadata),
                'output': output,
                'timestamp': datetime.now().isoformat()
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def interactive_mode(generator):
    """Interactive mode for testing."""
    print("\n" + "="*70)
    print("INTERACTIVE MODE - Claim Report Generation")
    print("="*70)
    print("\nEnter shipment metadata (or 'quit' to exit)")
    print("Format: shipment_id=XXX; damage_type=XXX; severity=XXX; ...")
    print("\nExample:")
    print("  shipment_id=SHP-12345; damage_type=dent; severity=high; vendor=TestCo; shipment_stage=transit")
    print("\n" + "-"*70)
    
    while True:
        print("\n📝 Input metadata:")
        user_input = input("> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Generate claim
            print("\n🔮 Generating claim report...")
            claim = generator.generate_claim(user_input)
            
            print("\n" + "="*70)
            print("GENERATED CLAIM REPORT")
            print("="*70)
            print(f"\n{claim}\n")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate claim reports from shipment metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction from command line
  python claim_report_generation.py --input "shipment_id=SHP-123; damage_type=dent; severity=high"
  
  # From JSON file
  python claim_report_generation.py --input_file input.json
  
  # Batch processing
  python claim_report_generation.py --batch inputs.jsonl --output outputs.jsonl
  
  # Interactive mode
  python claim_report_generation.py --interactive
        """
    )
    
    parser.add_argument('--input', type=str, help='Input metadata string')
    parser.add_argument('--input_file', type=str, help='JSON file with metadata dictionary')
    parser.add_argument('--batch', type=str, help='JSONL file with batch inputs')
    parser.add_argument('--output', type=str, help='Output file for batch processing')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model_path', type=str, help='Path to fine-tuned model')
    parser.add_argument('--max_length', type=int, default=256, help='Max output length')
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Load generator
    try:
        generator = ClaimReportGenerator(model_path=args.model_path)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return 1
    
    gen_kwargs = {
        'max_length': args.max_length,
        'num_beams': args.num_beams,
        'temperature': args.temperature,
    }
    
    # Interactive mode
    if args.interactive:
        interactive_mode(generator)
        return 0
    
    # Batch processing
    if args.batch:
        print(f"\n📦 Batch processing: {args.batch}")
        
        try:
            # Load inputs
            inputs = load_jsonl_batch(args.batch)
            print(f"   Loaded {len(inputs)} inputs")
            
            # Generate
            print("   Generating reports...")
            outputs = generator.batch_generate(inputs, **gen_kwargs)
            
            # Save outputs
            output_file = args.output or args.batch.replace('.jsonl', '_output.jsonl')
            save_jsonl_batch(outputs, inputs, output_file)
            
            print(f"   ✅ Saved {len(outputs)} outputs to {output_file}")
            
            # Show first few
            print("\n📋 Sample outputs:")
            for i, (inp, out) in enumerate(zip(inputs[:3], outputs[:3]), 1):
                print(f"\n--- Sample {i} ---")
                if isinstance(inp, dict):
                    print(f"Input: {json.dumps(inp)}")
                else:
                    print(f"Input: {inp}")
                print(f"Output: {out}")
            
        except Exception as e:
            print(f"\n❌ Error in batch processing: {e}")
            return 1
        
        return 0
    
    # Single input from file
    if args.input_file:
        print(f"\n📄 Loading input from: {args.input_file}")
        
        try:
            metadata = load_json_input(args.input_file)
            print(f"   Metadata: {json.dumps(metadata, indent=2)}")
            
            print("\n🔮 Generating claim report...")
            claim = generator.generate_from_dict(metadata, **gen_kwargs)
            
            print("\n" + "="*70)
            print("GENERATED CLAIM REPORT")
            print("="*70)
            print(f"\n{claim}\n")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
        
        return 0
    
    # Single input from command line
    if args.input:
        print(f"\n📝 Input: {args.input}")
        
        try:
            print("\n🔮 Generating claim report...")
            claim = generator.generate_claim(args.input, **gen_kwargs)
            
            print("\n" + "="*70)
            print("GENERATED CLAIM REPORT")
            print("="*70)
            print(f"\n{claim}\n")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
        
        return 0
    
    # No input provided - show help
    parser.print_help()
    print("\n💡 Tip: Use --interactive for testing multiple inputs")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
