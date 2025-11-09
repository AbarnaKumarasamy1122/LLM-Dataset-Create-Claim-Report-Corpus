# How to run
1. Activate virtualenv
2. Ensure data/metadata.csv exists
3. python scripts/enrich_metadata.py
4. python scripts/generate_reports.py
5. python scripts/sample_for_review.py   # produce manual_review template
6. Edit manual_review files and update data/review_log.csv
7. python scripts/export_jsonl.py
8. python scripts/sanity_checks.py
