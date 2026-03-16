#!/usr/bin/env bash
# scripts/local_setup_complete.sh
# Complete one-shot setup on your Ubuntu machine.
#
# Before running, export your tokens:
#   export KAGGLE_API_TOKEN="KGAT_..."
#   export GITHUB_TOKEN="github_pat_..."
#
# Or run: source scripts/set_tokens.sh  (gitignored file you create locally)

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

[[ -z "${KAGGLE_API_TOKEN:-}" ]] && { echo -e "${RED}ERROR: export KAGGLE_API_TOKEN first${NC}"; exit 1; }
[[ -z "${GITHUB_TOKEN:-}" ]]    && { echo -e "${RED}ERROR: export GITHUB_TOKEN first${NC}"; exit 1; }

KAGGLE_USER="ilanmeyrowitsch"
KAGGLE_DATA="${KAGGLE_DATA_DIR:-/home/ilan/kaggle/data}"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} RNA 3D Folding — Complete Local Setup  ${NC}"
echo -e "${CYAN}========================================${NC}"

echo -e "\n${CYAN}[1/6] Kaggle credentials...${NC}"
mkdir -p ~/.kaggle
echo "{\"username\":\"${KAGGLE_USER}\",\"key\":\"${KAGGLE_API_TOKEN}\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle -q --break-system-packages 2>/dev/null || pip install kaggle -q
echo -e "  ${GREEN}✓${NC} Done"

echo -e "\n${CYAN}[2/6] Linking competition data...${NC}"
bash scripts/setup_data_links.sh "${KAGGLE_DATA}"

echo -e "\n${CYAN}[3/6] Analyzing competition data...${NC}"
python3 scripts/analyze_data.py --data_dir "${KAGGLE_DATA}" 2>&1 | tail -15

echo -e "\n${CYAN}[4/6] Running tests...${NC}"
python3 -m pytest tests/ -q

echo -e "\n${CYAN}[5/6] Scoring baseline submission...${NC}"
python3 scripts/validate_submission.py \
    --submission "${KAGGLE_DATA}/sample_submission.csv" \
    --labels     "${KAGGLE_DATA}/validation_labels.csv" 2>&1 | tail -10

echo -e "\n${CYAN}[6/6] Pushing notebook to Kaggle + GitHub...${NC}"
bash scripts/push_to_kaggle.sh "Initial setup: $(date '+%Y-%m-%d')"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN} All done!                              ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Daily workflow:"
echo "  Edit src/ → bash scripts/push_to_kaggle.sh"
echo "  Submit:    bash scripts/submit_competition.sh"
