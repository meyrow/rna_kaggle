#!/usr/bin/env bash
# scripts/push_to_kaggle.sh
# Push notebook to Kaggle + push code to GitHub
# Run from project root: bash scripts/push_to_kaggle.sh

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

KAGGLE_USER="ilanmeyrowitsch"
KAGGLE_TOKEN="KGAT_207e7902a35c74689a20eb11f5606eba"
NOTEBOOK_ID="${KAGGLE_USER}/notebook-ilan"

echo -e "${CYAN}=== RNA 3D Folding — Push to Kaggle ===${NC}"

# ── 1. Set up Kaggle credentials ───────────────────────────────────
echo -e "\n${CYAN}[1/4] Setting up Kaggle credentials...${NC}"
mkdir -p ~/.kaggle
echo "{\"username\":\"${KAGGLE_USER}\",\"key\":\"${KAGGLE_TOKEN}\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Verify
kaggle config view 2>/dev/null || echo "kaggle not installed — run: pip install kaggle"
python3 -c "import kaggle; print('  Kaggle package: OK')" 2>/dev/null || \
    pip install kaggle --quiet

# ── 2. Regenerate notebook from latest src/ ────────────────────────
echo -e "\n${CYAN}[2/4] Regenerating notebook from src/...${NC}"
python3 build_notebook.py

# ── 3. Push notebook to Kaggle ─────────────────────────────────────
echo -e "\n${CYAN}[3/4] Pushing notebook to Kaggle...${NC}"
echo "  Notebook: ${NOTEBOOK_ID}"

# kernel-metadata.json must be in same dir as notebook
kaggle kernels push -p . 2>&1

echo -e "${GREEN}  Notebook pushed!${NC}"
echo "  View at: https://www.kaggle.com/code/${NOTEBOOK_ID}"

# ── 4. Check notebook status ───────────────────────────────────────
echo -e "\n${CYAN}[4/4] Checking notebook status...${NC}"
sleep 3
kaggle kernels status "${NOTEBOOK_ID}" 2>&1 || true

echo -e "\n${GREEN}=== Done ===${NC}"
echo ""
echo "Next:"
echo "  Watch run:    kaggle kernels status ${NOTEBOOK_ID}"
echo "  Pull output:  kaggle kernels output ${NOTEBOOK_ID} -p outputs/"
echo "  Submit:       kaggle competitions submit stanford-rna-3d-folding-2 \\"
echo "                  -f outputs/submission.csv -m 'Hybrid TBM+DeNovo pipeline'"
