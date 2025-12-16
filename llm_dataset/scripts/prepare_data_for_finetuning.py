"""
Week 5 - Task 1: Data Preparation for Fine-Tuning
Cleans, validates, and prepares JSONL datasets for LLM fine-tuning.

Input: data/llm_jsonl/{train,val,test}.jsonl
Output: model_training/{clean_train,clean_val,clean_test}.jsonl
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import hashlib

def load_jsonl(filepath):
    """Load JSONL file and return list of entries."""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append((line_num, entry))
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error at line {line_num}: {e}")
    return entries

def validate_entry(entry, line_num):
    """Validate that an entry has required fields and proper format."""
    issues = []
    
    # Check required fields
    if 'input' not in entry:
        issues.append(f"Missing 'input' field")
    elif not entry['input'] or not isinstance(entry['input'], str):
        issues.append(f"Invalid or empty 'input' field")
    
    if 'output' not in entry:
        issues.append(f"Missing 'output' field")
    elif not entry['output'] or not isinstance(entry['output'], str):
        issues.append(f"Invalid or empty 'output' field")
    
    # Check for placeholder text
    if 'output' in entry and entry['output']:
        output_text = entry['output'].lower()
        placeholders = ['{', '}', '<', '>', 'placeholder', 'todo', 'fixme']
        for placeholder in placeholders:
            if placeholder in output_text and placeholder in ['{', '}', '<', '>']:
                issues.append(f"Potential template placeholder in output: {placeholder}")
                break
    
    # Check minimum length
    if 'input' in entry and len(entry['input']) < 20:
        issues.append(f"Input too short ({len(entry['input'])} chars)")
    if 'output' in entry and len(entry['output']) < 30:
        issues.append(f"Output too short ({len(entry['output'])} chars)")
    
    return issues

def compute_hash(entry):
    """Compute hash of input+output for duplicate detection."""
    text = entry.get('input', '') + '|||' + entry.get('output', '')
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def clean_text(text):
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Fix common encoding issues
    text = text.replace('â‚¹', '₹')
    return text.strip()

def process_dataset(input_path, output_path, dataset_name):
    """Process and clean a single dataset file."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}...")
    print(f"{'='*60}")
    
    # Load data
    entries = load_jsonl(input_path)
    print(f"✓ Loaded {len(entries)} entries from {input_path}")
    
    # Validate entries
    valid_entries = []
    invalid_entries = []
    seen_hashes = set()
    duplicate_count = 0
    
    for line_num, entry in entries:
        # Validate
        issues = validate_entry(entry, line_num)
        
        if issues:
            invalid_entries.append((line_num, entry, issues))
            print(f"⚠️  Line {line_num}: {', '.join(issues)}")
            continue
        
        # Clean text
        entry['input'] = clean_text(entry['input'])
        entry['output'] = clean_text(entry['output'])
        
        # Check for duplicates
        entry_hash = compute_hash(entry)
        if entry_hash in seen_hashes:
            duplicate_count += 1
            continue
        
        seen_hashes.add(entry_hash)
        valid_entries.append(entry)
    
    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Total entries: {len(entries)}")
    print(f"   Valid entries: {len(valid_entries)}")
    print(f"   Invalid entries: {len(invalid_entries)}")
    print(f"   Duplicates removed: {duplicate_count}")
    print(f"   Success rate: {len(valid_entries)/len(entries)*100:.1f}%")
    
    # Analyze input/output lengths
    input_lengths = [len(e['input']) for e in valid_entries]
    output_lengths = [len(e['output']) for e in valid_entries]
    
    print(f"\n📏 Length Analysis:")
    print(f"   Input - Min: {min(input_lengths)}, Max: {max(input_lengths)}, Avg: {sum(input_lengths)/len(input_lengths):.0f}")
    print(f"   Output - Min: {min(output_lengths)}, Max: {max(output_lengths)}, Avg: {sum(output_lengths)/len(output_lengths):.0f}")
    
    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in valid_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Saved {len(valid_entries)} clean entries to {output_path}")
    
    return {
        'total': len(entries),
        'valid': len(valid_entries),
        'invalid': len(invalid_entries),
        'duplicates': duplicate_count,
        'input_lengths': input_lengths,
        'output_lengths': output_lengths
    }

def analyze_content_patterns(entries):
    """Analyze patterns in the dataset."""
    damage_types = defaultdict(int)
    severity_levels = defaultdict(int)
    
    for entry in entries:
        input_text = entry.get('input', '')
        
        # Extract damage types
        if 'damage_type=' in input_text:
            parts = input_text.split('damage_type=')
            if len(parts) > 1:
                damage_type = parts[1].split(';')[0].strip()
                damage_types[damage_type] += 1
        
        # Extract severity
        if 'severity=' in input_text:
            parts = input_text.split('severity=')
            if len(parts) > 1:
                severity = parts[1].split(';')[0].strip()
                severity_levels[severity] += 1
    
    return damage_types, severity_levels

def generate_report(stats):
    """Generate a summary report."""
    report_lines = [
        "\n" + "="*60,
        "DATA PREPARATION SUMMARY",
        "="*60,
        "",
        f"Total entries processed: {sum(s['total'] for s in stats.values())}",
        f"Total valid entries: {sum(s['valid'] for s in stats.values())}",
        f"Total invalid entries: {sum(s['invalid'] for s in stats.values())}",
        f"Total duplicates removed: {sum(s['duplicates'] for s in stats.values())}",
        "",
        "Dataset Breakdown:",
        f"  Training:   {stats['train']['valid']:,} entries",
        f"  Validation: {stats['val']['valid']:,} entries",
        f"  Test:       {stats['test']['valid']:,} entries",
        "",
        "✅ All datasets are ready for fine-tuning!",
        "="*60,
    ]
    
    report = "\n".join(report_lines)
    print(report)
    
    # Save report
    report_path = Path("model_training/data_preparation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Report saved to {report_path}")

def main():
    """Main execution function."""
    print("\n🚀 Starting Data Preparation for Fine-Tuning...")
    print("="*60)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "llm_jsonl"
    output_dir = base_dir / "model_training"
    
    datasets = {
        'train': ('train.jsonl', 'clean_train.jsonl'),
        'val': ('val.jsonl', 'clean_val.jsonl'),
        'test': ('test.jsonl', 'clean_test.jsonl')
    }
    
    stats = {}
    
    # Process each dataset
    for name, (input_file, output_file) in datasets.items():
        input_path = input_dir / input_file
        output_path = output_dir / output_file
        
        if not input_path.exists():
            print(f"⚠️  Warning: {input_path} not found, skipping...")
            continue
        
        stats[name] = process_dataset(input_path, output_path, name.upper())
    
    # Generate summary report
    if stats:
        generate_report(stats)
    
    print("\n✨ Data preparation complete!")

if __name__ == "__main__":
    main()
