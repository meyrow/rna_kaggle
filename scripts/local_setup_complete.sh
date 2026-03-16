#!/usr/bin/env bash
# scripts/local_setup_complete.sh
# Complete one-shot setup on your local Ubuntu machine
# Run: bash scripts/local_setup_complete.sh

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

KAGGLE_USER="ilanmeyrowitsch"
KAGGLE_TOKEN="KGAT_207e7902a35c74689a20eb11f5606eba"
KAGGLE_DATA="/home/ilan/kaggle/data"
GITHUB_REPO="https://github.com/meyrow/rna_kaggle.git"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} RNA 3D Folding — Complete Local Setup  ${NC}"
echo -e "${CYAN}========================================${NC}"

# ── Kaggle credentials ─────────────────────────────────────────────
echo -e "\n${CYAN}[1/6] Kaggle credentials...${NC}"
mkdir -p ~/.kaggle
echo "{\"username\":\"${KAGGLE_USER}\",\"key\":\"${KAGGLE_TOKEN}\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle --quiet --break-system-packages 2>/dev/null || pip install kaggle --quiet
echo -e "  ${GREEN}✓${NC} Kaggle credentials set"

# ── Data symlinks ──────────────────────────────────────────────────
echo -e "\n${CYAN}[2/6] Linking competition data...${NC}"
bash scripts/setup_data_links.sh "${KAGGLE_DATA}"

# ── Run data analysis ──────────────────────────────────────────────
echo -e "\n${CYAN}[3/6] Analyzing competition data...${NC}"
python3 scripts/analyze_data.py --data_dir "${KAGGLE_DATA}" 2>&1 | tail -20

# ── Run tests ──────────────────────────────────────────────────────
echo -e "\n${CYAN}[4/6] Running tests...${NC}"
python3 -m pytest tests/ -q 2>&1

# ── Validate sample submission ─────────────────────────────────────
echo -e "\n${CYAN}[5/6] Scoring sample submission (baseline)...${NC}"
python3 scripts/validate_submission.py \
    --submission "${KAGGLE_DATA}/sample_submission.csv" \
    --labels "${KAGGLE_DATA}/validation_labels.csv" 2>&1 | tail -15

# ── Push notebook to Kaggle ────────────────────────────────────────
echo -e "\n${CYAN}[6/6] Pushing notebook to Kaggle...${NC}"
python3 build_notebook.py
kaggle kernels push -p . 2>&1
echo -e "  ${GREEN}✓${NC} Notebook pushed to: https://www.kaggle.com/code/${KAGGLE_USER}/notebook-ilan"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN} Setup complete!                        ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Daily workflow:"
echo "  Edit src/ → python3 build_notebook.py → bash scripts/push_to_kaggle.sh"
echo ""
echo "To submit after notebook runs on Kaggle:"
echo "  bash scripts/submit_competition.sh"
