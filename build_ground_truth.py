"""Build ground truth eval_set_valid.csv by matching eval prefixes to variant sequences."""
import csv
from collections import defaultdict

# Load all full sequences, indexed by (family, first 5 steps as key)
print("Loading variant sequences...")
all_seqs = {}  # family -> {prefix_key: [full_seq, ...]}
for fam, fname in [('MOSFET', 'data/raw/infineon/MOSFET/MOSFET_variants.csv'),
                   ('IGBT', 'data/raw/infineon/IGBT/IGBT_variants.csv'),
                   ('IC', 'data/raw/infineon/IC/IC_variants.csv')]:
    seqs_by_id = defaultdict(list)
    with open(fname) as f:
        for row in csv.DictReader(f):
            seqs_by_id[row['SEQUENCE_ID']].append(row['STEP'])
    
    # Index by first 5 steps for fast lookup
    indexed = defaultdict(list)
    for seq in seqs_by_id.values():
        key = tuple(seq[:5])
        indexed[key].append(seq)
    all_seqs[fam] = indexed
    print(f"  {fam}: {len(seqs_by_id)} sequences, {len(indexed)} unique prefixes")

# Load eval_input_valid.csv
print("\nMatching eval examples to full sequences...")
with open('data/participant_files/eval_input_valid.csv', encoding='utf-8-sig') as f:
    evals = list(csv.DictReader(f))

matched = 0
unmatched_ids = []
gt_rows = []

for ex in evals:
    eid = ex['EXAMPLE_ID'].strip()
    family = ex['FAMILY'].strip()
    partial = [s.strip() for s in ex['PARTIAL_SEQUENCE'].split('|') if s.strip()]
    prefix_len = len(partial)
    
    # Use first 5 steps as lookup key
    key = tuple(partial[:5])
    candidates = all_seqs.get(family, {}).get(key, [])
    
    found = False
    for full_seq in candidates:
        if full_seq[:prefix_len] == partial:
            next_step = full_seq[prefix_len] if prefix_len < len(full_seq) else ''
            gt_rows.append({
                'EXAMPLE_ID': eid,
                'FAMILY': family,
                'COMPLETION_FRACTION': ex['COMPLETION_FRACTION'].strip(),
                'PARTIAL_SEQUENCE': '|'.join(partial),
                'FULL_SEQUENCE': '|'.join(full_seq),
                'NEXT_STEP': next_step,
            })
            found = True
            matched += 1
            break
    if not found:
        unmatched_ids.append(eid)

print(f"\nMatched: {matched}/{len(evals)}, Unmatched: {len(unmatched_ids)}")
if unmatched_ids[:5]:
    print(f"  Unmatched examples: {unmatched_ids[:5]}")

if gt_rows:
    with open('data/participant_files/eval_set_valid.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'EXAMPLE_ID', 'FAMILY', 'COMPLETION_FRACTION',
            'PARTIAL_SEQUENCE', 'FULL_SEQUENCE', 'NEXT_STEP'])
        writer.writeheader()
        writer.writerows(gt_rows)
    print(f"Wrote data/participant_files/eval_set_valid.csv ({len(gt_rows)} rows)")
