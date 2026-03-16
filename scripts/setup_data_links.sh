#!/usr/bin/env bash
# scripts/setup_data_links.sh
# Create symlinks from your kaggle/data/ folder into the project's data/ tree.
# Run once from the project root after cloning.
#
# Usage:
#   bash scripts/setup_data_links.sh [kaggle_data_dir]
#   bash scripts/setup_data_links.sh /home/ilan/kaggle/data   # default

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

KAGGLE_DATA="${1:-/home/ilan/kaggle/data}"

echo -e "${CYAN}=== Setting up data symlinks ===${NC}"
echo "Kaggle data dir: $KAGGLE_DATA"

if [ ! -d "$KAGGLE_DATA" ]; then
    echo -e "${YELLOW}WARNING: $KAGGLE_DATA not found. Update the path.${NC}"
    exit 1
fi

mkdir -p data/raw data/pdb_cache/structures

# ── CSV files ──────────────────────────────────────────────────────
for f in test_sequences.csv validation_sequences.csv \
          validation_labels.csv sample_submission.csv \
          train_sequences.csv train_labels.csv; do
    src="$KAGGLE_DATA/$f"
    dst="data/raw/$f"
    if [ -f "$src" ]; then
        ln -sf "$src" "$dst"
        echo -e "  ${GREEN}✓${NC} data/raw/$f → $src"
    else
        echo -e "  ${YELLOW}✗${NC} Not found: $src"
    fi
done

# ── PDB_RNA structures ─────────────────────────────────────────────
PDB_RNA="$KAGGLE_DATA/PDB_RNA"
if [ -d "$PDB_RNA" ]; then
    # Symlink the whole directory
    ln -sfn "$PDB_RNA" data/pdb_cache/structures
    N=$(ls "$PDB_RNA" | wc -l)
    echo -e "  ${GREEN}✓${NC} data/pdb_cache/structures → $PDB_RNA ($N files)"
else
    echo -e "  ${YELLOW}✗${NC} PDB_RNA/ not found at $PDB_RNA"
fi

# ── Extra data ─────────────────────────────────────────────────────
EXTRA="$KAGGLE_DATA/extra"
if [ -d "$EXTRA" ]; then
    ln -sfn "$EXTRA" data/extra
    echo -e "  ${GREEN}✓${NC} data/extra → $EXTRA"
fi

echo -e "\n${GREEN}Data links ready.${NC}"
echo "Next: bash scripts/build_pdb_cache.sh"
