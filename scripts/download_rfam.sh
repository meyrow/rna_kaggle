#!/usr/bin/env bash
# scripts/download_rfam.sh — Download Rfam covariance model database
# Used by family_classifier.py for cmscan family detection
# Run from project root: bash scripts/download_rfam.sh

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; NC='\033[0m'
RFAM_DIR="data/rfam"

echo -e "${CYAN}=== Downloading Rfam database ===${NC}"
mkdir -p "$RFAM_DIR"

# Rfam.cm.gz — all covariance models (~500MB compressed)
RFAM_URL="https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz"
echo "Downloading Rfam.cm.gz (~500MB)..."
wget -q --show-progress -O "$RFAM_DIR/Rfam.cm.gz" "$RFAM_URL"
gunzip -f "$RFAM_DIR/Rfam.cm.gz"

# Press the CM database for fast cmscan access
echo "Pressing Rfam database (cmpress)..."
cmpress "$RFAM_DIR/Rfam.cm"

echo -e "${GREEN}Rfam database ready at $RFAM_DIR/Rfam.cm${NC}"
ls -lh "$RFAM_DIR/"
