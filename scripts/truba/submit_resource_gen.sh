#!/bin/bash
#SBATCH --job-name=resource_gen
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --output=$SCRATCH_DIR/logs/resource_gen_%j.out
#SBATCH --error=$SCRATCH_DIR/logs/resource_gen_%j.err

set -euo pipefail

PROJECT_DIR=$PROJECT_DIR

echo "=== Resource Generation (Gen 0 completion + Gen 1) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Started: $(date)"

module load comp/python/miniconda3
source "$PROJECT_DIR/.venv/bin/activate"

mkdir -p $SCRATCH_DIR/logs

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

# ------------------------------------------------------------------
# Phase 1: Complete Gen 0 (Zeyrek bulk + tier re-evaluation)
# DB already has BOUN + IMST + UniMorph from partial run on login node
# ------------------------------------------------------------------
echo ""
echo "=== Phase 1: Completing Gen 0 (Zeyrek + tiers) ==="
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from aksu.resource.schema import MorphDatabase, MorphEntry
from aksu.resource.importers.zeyrek_bulk import analyze_bulk
from aksu.resource.quality_check import tier_from_entries

db = MorphDatabase('data/resource/tr_gold_morph.db')

# Check if Zeyrek already ran
stats = db.get_stats()
by_source = stats['by_source']
if 'zeyrek' in by_source:
    print(f'Zeyrek already present ({by_source[\"zeyrek\"]:,} entries). Skipping bulk analysis.')
else:
    all_surfaces = db.get_all_surfaces()
    print(f'Running Zeyrek on {len(all_surfaces):,} unique surfaces ...')
    zeyrek_results = analyze_bulk(all_surfaces)

    zeyrek_entries = []
    for surface, canonical in zeyrek_results.items():
        if canonical is None:
            continue
        parts = canonical.split()
        lemma = parts[0] if parts else surface
        pos = 'NOUN'
        for part in parts[1:]:
            if part in ('+Noun', '+Verb', '+Adj', '+Adv'):
                pos = part.lstrip('+').upper()
                break
        zeyrek_entries.append(MorphEntry(
            surface=surface, lemma=lemma, canonical_tags=canonical,
            pos=pos, source='zeyrek', confidence=0.9, frequency=1, tier='bronze',
        ))
    db.bulk_insert(zeyrek_entries)
    print(f'Zeyrek: {len(zeyrek_entries):,} entries inserted')

# Re-evaluate tiers
print('Re-evaluating tiers by cross-source agreement ...')
all_surfaces = db.get_all_surfaces()
updated = 0
for surface in all_surfaces:
    entries = db.query_surface(surface)
    if len(entries) <= 1:
        continue
    tier, agreement = tier_from_entries(entries)
    for entry in entries:
        db.update_tier(surface, entry.source, tier, agreement)
    updated += 1
print(f'Updated tiers for {updated:,} multi-source surfaces')

stats = db.get_stats()
print(f'Gen 0 final: {stats[\"total\"]:,} entries, {stats[\"unique_surfaces\"]:,} surfaces')
print(f'  By tier:   {stats[\"by_tier\"]}')
print(f'  By source: {stats[\"by_source\"]}')
db.close()
"

echo ""
echo "=== Phase 2: Exporting training data ==="
python -c "
from aksu.resource.training_bridge import export_training_data
export_training_data('data/resource/tr_gold_morph.db', 'data/resource/training_export.jsonl')
"

echo ""
echo "=== Phase 3: Generation 1 — Wikipedia ==="
python src/resource/generation_runner.py --generation 1

echo ""
echo "Finished: $(date)"
