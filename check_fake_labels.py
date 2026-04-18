#!/usr/bin/env python3
"""
Check for fake labels in the training data
Fake labels might have:
- Unusual tau values (should be 0-1)
- Unusual rho values (should be 0-1)
- Unusual rs values (should be 0-1)
- Unusual eval_score values (should be reasonable centipawn range)
- Missing or null values
"""
import json
from pathlib import Path
from collections import defaultdict

data_path = Path("../downloaded_data/full_data_770k.jsonl")

print("Loading data...")
lines = []
with open(data_path, 'r') as f:
    for line in f:
        if line.strip():
            lines.append(json.loads(line))

print(f"Loaded {len(lines)} lines")

# Check for fake labels
print("\nChecking for fake labels...")

fake_count = 0
issues = defaultdict(int)

for i, data in enumerate(lines):
    has_issue = False
    
    # Check eval_score
    eval_score = data.get('eval_score')
    if eval_score is None:
        issues['missing_eval'] += 1
        has_issue = True
    elif not isinstance(eval_score, (int, float)):
        issues['invalid_eval_type'] += 1
        has_issue = True
    elif abs(eval_score) > 10000:  # Unreasonable eval
        issues['extreme_eval'] += 1
        has_issue = True
    
    # Check tau
    tau = data.get('tau')
    if tau is None:
        issues['missing_tau'] += 1
        has_issue = True
    elif not isinstance(tau, (int, float)):
        issues['invalid_tau_type'] += 1
        has_issue = True
    elif tau < 0 or tau > 1:
        issues['invalid_tau_range'] += 1
        has_issue = True
    
    # Check rho
    rho = data.get('rho')
    if rho is None:
        issues['missing_rho'] += 1
        has_issue = True
    elif not isinstance(rho, (int, float)):
        issues['invalid_rho_type'] += 1
        has_issue = True
    elif rho < 0 or rho > 1:
        issues['invalid_rho_range'] += 1
        has_issue = True
    
    # Check rs
    rs = data.get('rs')
    if rs is None:
        issues['missing_rs'] += 1
        has_issue = True
    elif not isinstance(rs, (int, float)):
        issues['invalid_rs_type'] += 1
        has_issue = True
    elif rs < 0 or rs > 1:
        issues['invalid_rs_range'] += 1
        has_issue = True
    
    if has_issue:
        fake_count += 1

print(f"\n{'='*60}")
print(f"SUMMARY: Fake Label Detection")
print(f"{'='*60}")
print(f"Total lines: {len(lines)}")
print(f"Lines with issues: {fake_count} ({fake_count/len(lines)*100:.2f}%)")

if issues:
    print(f"\nIssues found:")
    for issue, count in sorted(issues.items(), key=lambda x: x[1], reverse=True):
        print(f"  {issue}: {count} ({count/len(lines)*100:.2f}%)")
else:
    print("\nNo issues found - all labels appear valid")

# Show sample of values
print(f"\n{'='*60}")
print(f"Label Value Statistics")
print(f"{'='*60}")

eval_scores = [d.get('eval_score') for d in lines if d.get('eval_score') is not None]
taus = [d.get('tau') for d in lines if d.get('tau') is not None]
rhos = [d.get('rho') for d in lines if d.get('rho') is not None]
rss = [d.get('rs') for d in lines if d.get('rs') is not None]

if eval_scores:
    print(f"\neval_score:")
    print(f"  Min: {min(eval_scores):.2f}")
    print(f"  Max: {max(eval_scores):.2f}")
    print(f"  Mean: {sum(eval_scores)/len(eval_scores):.2f}")

if taus:
    print(f"\ntau:")
    print(f"  Min: {min(taus):.4f}")
    print(f"  Max: {max(taus):.4f}")
    print(f"  Mean: {sum(taus)/len(taus):.4f}")

if rhos:
    print(f"\nrho:")
    print(f"  Min: {min(rhos):.4f}")
    print(f"  Max: {max(rhos):.4f}")
    print(f"  Mean: {sum(rhos)/len(rhos):.4f}")

if rss:
    print(f"\nrs:")
    print(f"  Min: {min(rss):.4f}")
    print(f"  Max: {max(rss):.4f}")
    print(f"  Mean: {sum(rss)/len(rss):.4f}")
