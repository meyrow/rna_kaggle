#!/usr/bin/env bash
# scripts/download_models.sh — Download RibonanzaNet2 and Protenix checkpoints
# Requires: Kaggle API credentials in ~/.kaggle/kaggle.json
# Run from project root: bash scripts/download_models.sh

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}=== Downloading model checkpoints ===${NC}"

# ── Check Kaggle API ────────────────────────────────────────────────
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo -e "${YELLOW}Kaggle API credentials not found at ~/.kaggle/kaggle.json${NC}"
    echo "Get your API key from: https://www.kaggle.com/settings (Account → API → Create New Token)"
    echo "Then: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"

# ── RibonanzaNet2 ────────────────────────────────────────────────────
echo -e "\n${CYAN}[1/2] Downloading RibonanzaNet2 (~400MB)...${NC}"
mkdir -p models/ribonanzanet2
if [ ! -f "models/ribonanzanet2/pytorch_model_fsdp.bin" ]; then
    kaggle models instances versions download \
        shujun717/ribonanzanet2/pyTorch/alpha/1 \
        -p models/ribonanzanet2/ --untar
    echo -e "${GREEN}  RibonanzaNet2 downloaded${NC}"
else
    echo "  RibonanzaNet2 already present"
fi

# ── Protenix checkpoint ──────────────────────────────────────────────
echo -e "\n${CYAN}[2/2] Downloading Protenix base checkpoint (~1.5GB)...${NC}"
mkdir -p models/protenix
PROTENIX_URL="https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt"
if [ ! -f "models/protenix/protenix_base_default_v0.5.0.pt" ]; then
    wget -q --show-progress -O "models/protenix/protenix_base_default_v0.5.0.pt" "$PROTENIX_URL"
    echo -e "${GREEN}  Protenix downloaded${NC}"
else
    echo "  Protenix already present"
fi

# ── Kaggle competition MSA data ──────────────────────────────────────
echo -e "\n${CYAN}[+] Downloading precomputed MSA data (MSA_v2, ~15GB)...${NC}"
echo -e "${YELLOW}  WARNING: This is large (~15GB). Skipping by default.${NC}"
echo "  To download manually:"
echo "    kaggle datasets download rhijudas/clone-of-stanford-rna-3d-modeling-competition-data"
echo "    unzip *.zip 'MSA_v2/*' -d data/msa/"

# ── Verify ────────────────────────────────────────────────────────────
echo -e "\n${GREEN}=== Model download complete ===${NC}"
echo "Checksums:"
for f in models/ribonanzanet2/pytorch_model_fsdp.bin models/protenix/protenix_base_default_v0.5.0.pt; do
    if [ -f "$f" ]; then
        echo "  ✓ $f ($(du -sh "$f" | cut -f1))"
    else
        echo "  ✗ $f — NOT FOUND"
    fi
done
