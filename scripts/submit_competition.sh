#!/usr/bin/env bash
# scripts/submit_competition.sh
# Pull output from Kaggle notebook and submit to competition
# Run AFTER notebook has finished running on Kaggle

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; NC='\033[0m'

KAGGLE_USER="ilanmeyrowitsch"
NOTEBOOK_ID="${KAGGLE_USER}/notebook-ilan"
COMPETITION="stanford-rna-3d-folding-2"
MESSAGE="${1:-Hybrid TBM+DeNovo pipeline v$(date +%Y%m%d-%H%M)}"

echo -e "${CYAN}=== Competition Submission ===${NC}"

# Check notebook status
echo -e "\n${CYAN}[1/3] Checking notebook status...${NC}"
STATUS=$(kaggle kernels status "${NOTEBOOK_ID}" 2>&1)
echo "$STATUS"

if echo "$STATUS" | grep -q "running"; then
    echo -e "\nNotebook still running. Wait for it to complete, then re-run this script."
    exit 1
fi

# Pull output
echo -e "\n${CYAN}[2/3] Pulling submission.csv from Kaggle...${NC}"
mkdir -p outputs
kaggle kernels output "${NOTEBOOK_ID}" -p outputs/
ls -lh outputs/

# Submit
SUBMISSION_FILE="outputs/submission.csv"
if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "ERROR: $SUBMISSION_FILE not found in notebook output"
    echo "Check: kaggle kernels output ${NOTEBOOK_ID} -p outputs/"
    exit 1
fi

echo -e "\n${CYAN}[3/3] Submitting to competition...${NC}"
kaggle competitions submit "${COMPETITION}" \
    -f "${SUBMISSION_FILE}" \
    -m "${MESSAGE}"

echo -e "\n${GREEN}=== Submitted! ===${NC}"
echo "Check leaderboard: https://www.kaggle.com/competitions/${COMPETITION}/leaderboard"
