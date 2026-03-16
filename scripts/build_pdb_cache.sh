#!/usr/bin/env bash
# scripts/build_pdb_cache.sh — Build local PDB RNA template search index
# One-time setup: ~1-2h, requires ~15GB disk space
# Usage:
#   bash scripts/build_pdb_cache.sh
#   bash scripts/build_pdb_cache.sh --pdb_dir /home/ilan/kaggle/data/PDB_RNA
#
# After running setup_data_links.sh, data/pdb_cache/structures is already
# symlinked to /home/ilan/kaggle/data/PDB_RNA — no --pdb_dir needed.

set -euo pipefail
# Parse optional --pdb_dir argument
PDB_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --pdb_dir) PDB_DIR_OVERRIDE="$2"; shift 2;;
        *) shift;;
    esac
done

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PDB_DIR="data/pdb_cache/structures"
SEQ_FA="data/pdb_cache/pdb_rna_seqs.fa"
MMSEQS_DB="data/pdb_cache/pdb_rna_mmseqs2"
C1_CACHE="data/pdb_cache/pdb_c1_coords.pkl"

echo -e "${CYAN}=== Building PDB RNA template search index ===${NC}"
mkdir -p "$PDB_DIR" data/pdb_cache

# ── Step 1: Download RNA-containing PDB structures ──────────────────
echo -e "\n${CYAN}[1/4] Downloading RNA PDB structures...${NC}"
echo "  Fetching RNA-containing PDB IDs from RCSB..."
# Query RCSB for all structures containing RNA
curl -s "https://search.rcsb.org/rcsbsearch/v2/query" \
    -H "Content-Type: application/json" \
    -d '{
      "query": {"type": "terminal", "service": "text",
                "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                               "operator": "greater", "value": 0}},
      "return_type": "entry",
      "request_options": {"paginate": {"start": 0, "rows": 50000}}
    }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
ids = [r['identifier'] for r in data.get('result_set', [])]
print('\n'.join(ids))
" > data/pdb_cache/rna_pdb_ids.txt 2>/dev/null || {
    echo -e "${YELLOW}  RCSB API query failed. Using pre-curated list.${NC}"
    echo "  Download manually: https://www.rcsb.org/search?request=..."
}

N_IDS=$(wc -l < data/pdb_cache/rna_pdb_ids.txt 2>/dev/null || echo 0)
echo "  Found $N_IDS RNA structure IDs"

# Download PDB files (batch, rsync from RCSB mirror)
echo "  Downloading PDB files via rsync (this will take a while)..."
echo -e "${YELLOW}  Tip: Consider using the Kaggle training dataset instead (already RNA-curated)${NC}"
echo "    kaggle datasets download rhijudas/stanford-rna-3d-folding-all-atom-train-data"

# ── Step 2: Extract C1' sequences for MMseqs2 DB ────────────────────
echo -e "\n${CYAN}[2/4] Extracting C1' sequences for MMseqs2...${NC}"
python3 - <<'EOF'
from pathlib import Path
from src.utils.pdb_parser import extract_c1_coords

pdb_dir = Path("data/pdb_cache/structures")
out_fa = Path("data/pdb_cache/pdb_rna_seqs.fa")
pdb_files = list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.ent"))

print(f"  Processing {len(pdb_files)} PDB files...")
count = 0
with open(out_fa, "w") as f:
    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.upper()
        try:
            coords, seq = extract_c1_coords(str(pdb_path))
            if len(seq) >= 10:  # skip very short fragments
                f.write(f">{pdb_id}_A\n{seq}\n")
                count += 1
        except Exception:
            pass
print(f"  Written {count} sequences to {out_fa}")
EOF

# ── Step 3: Build MMseqs2 database ──────────────────────────────────
echo -e "\n${CYAN}[3/4] Building MMseqs2 search database...${NC}"
if [ -f "$SEQ_FA" ] && [ "$(wc -l < $SEQ_FA)" -gt 10 ]; then
    mmseqs createdb "$SEQ_FA" "$MMSEQS_DB"
    mmseqs createindex "$MMSEQS_DB" /tmp/mmseqs_tmp --threads 8
    echo -e "${GREEN}  MMseqs2 DB built at $MMSEQS_DB${NC}"
else
    echo -e "${YELLOW}  No sequences found. Build structures directory first.${NC}"
fi

# ── Step 4: Build C1' coordinate pickle cache ────────────────────────
echo -e "\n${CYAN}[4/4] Building C1' coordinate cache...${NC}"
python3 - <<'EOF'
from src.utils.pdb_parser import build_pdb_c1_cache
build_pdb_c1_cache("data/pdb_cache/structures", "data/pdb_cache/pdb_c1_coords.pkl")
EOF

echo -e "\n${GREEN}=== PDB cache build complete ===${NC}"
echo "Files:"
for f in "$MMSEQS_DB.dbtype" "$C1_CACHE"; do
    [ -f "$f" ] && echo "  ✓ $f" || echo "  ✗ $f"
done
