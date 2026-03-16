#!/usr/bin/env bash
# scripts/push_to_kaggle.sh
# Regenerate notebook → push to Kaggle + GitHub
#
# Requires env vars:
#   export KAGGLE_API_TOKEN="KGAT_..."
#   export GITHUB_TOKEN="github_pat_..."
#
# Usage: bash scripts/push_to_kaggle.sh [commit message]

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

MSG="${1:-Update notebook: $(date '+%Y-%m-%d %H:%M')}"

# ── Check env vars ─────────────────────────────────────────────────
[[ -z "${KAGGLE_API_TOKEN:-}" ]] && { echo -e "${RED}ERROR: export KAGGLE_API_TOKEN first${NC}"; exit 1; }
[[ -z "${GITHUB_TOKEN:-}" ]]    && { echo -e "${RED}ERROR: export GITHUB_TOKEN first${NC}"; exit 1; }

KAGGLE_USER="ilanmeyrowitsch"
NOTEBOOK_ID="${KAGGLE_USER}/notebook-ilan"

echo -e "${CYAN}=== RNA 3D Folding — Push to Kaggle + GitHub ===${NC}"

# ── 1. Kaggle credentials ──────────────────────────────────────────
echo -e "\n${CYAN}[1/5] Kaggle credentials...${NC}"
mkdir -p ~/.kaggle
echo "{\"username\":\"${KAGGLE_USER}\",\"key\":\"${KAGGLE_API_TOKEN}\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
python3 -c "import kaggle" 2>/dev/null || pip install kaggle --break-system-packages -q

# ── 2. Regenerate notebook ─────────────────────────────────────────
echo -e "\n${CYAN}[2/5] Regenerating notebook from src/...${NC}"
python3 build_notebook.py

# ── 3. Push to Kaggle ──────────────────────────────────────────────
echo -e "\n${CYAN}[3/5] Pushing notebook to Kaggle...${NC}"
kaggle kernels push -p .
echo -e "  ${GREEN}✓${NC} https://www.kaggle.com/code/${NOTEBOOK_ID}"

# ── 4. Run tests ───────────────────────────────────────────────────
echo -e "\n${CYAN}[4/5] Running tests...${NC}"
# Install pytest if missing
python3 -c "import pytest" 2>/dev/null || \
    pip install pytest --break-system-packages -q 2>/dev/null || \
    pip install pytest -q

python3 -m pytest tests/ -q --tb=short
echo -e "  ${GREEN}✓${NC} All tests passed"

# ── 5. Push to GitHub ──────────────────────────────────────────────
echo -e "\n${CYAN}[5/5] Pushing to GitHub...${NC}"
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/meyrow/rna_kaggle.git"
git add -A
git diff --staged --quiet || git commit -m "${MSG}"
git push origin main
echo -e "  ${GREEN}✓${NC} https://github.com/meyrow/rna_kaggle"

echo -e "\n${GREEN}=== Done! ===${NC}"
echo ""
echo "Watch Kaggle run:    kaggle kernels status ${NOTEBOOK_ID}"
echo "Get output:          kaggle kernels output ${NOTEBOOK_ID} -p outputs/"
echo "Submit when done:    bash scripts/submit_competition.sh"
