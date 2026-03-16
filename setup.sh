#!/usr/bin/env bash
# setup.sh — One-shot environment setup for RNA 3D Folding Pipeline
# Target: Ubuntu 22.04/24.04, RTX 4060 (8GB VRAM), i9-13980HX, 32GB RAM
# Run: bash setup.sh

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}=== RNA 3D Folding Pipeline — Environment Setup ===${NC}"
echo "Target: Ubuntu + RTX 4060 (CUDA 12.1), Python 3.10"

# ── 0. Prerequisites ────────────────────────────────────────────────
echo -e "\n${CYAN}[0/7] Checking prerequisites...${NC}"
command -v conda >/dev/null 2>&1 || {
    echo -e "${YELLOW}conda not found. Please install Miniconda first:${NC}"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
}
command -v nvcc >/dev/null 2>&1 || echo -e "${YELLOW}Warning: nvcc not found. CUDA may not be installed.${NC}"

# ── 1. Create conda environment ─────────────────────────────────────
echo -e "\n${CYAN}[1/7] Creating conda environment 'rna_folding' (Python 3.10)...${NC}"
conda create -n rna_folding python=3.10 -y || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rna_folding

# ── 2. Install PyTorch with CUDA 12.1 ───────────────────────────────
echo -e "\n${CYAN}[2/7] Installing PyTorch 2.2 + CUDA 12.1...${NC}"
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); \
           print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" || true

# ── 3. Install ViennaRNA (secondary structure) ───────────────────────
echo -e "\n${CYAN}[3/7] Installing ViennaRNA...${NC}"
conda install -c bioconda viennarna -y || {
    echo -e "${YELLOW}bioconda install failed, trying pip...${NC}"
    pip install viennarna || echo -e "${YELLOW}ViennaRNA not available via pip. Install manually.${NC}"
}

# ── 4. Install Infernal (Rfam cmscan) ───────────────────────────────
echo -e "\n${CYAN}[4/7] Installing Infernal (cmscan for Rfam)...${NC}"
conda install -c bioconda infernal -y || {
    echo -e "${YELLOW}Trying apt...${NC}"
    sudo apt-get install -y infernal || echo -e "${YELLOW}Install infernal manually.${NC}"
}

# ── 5. Install MMseqs2 (template search) ────────────────────────────
echo -e "\n${CYAN}[5/7] Installing MMseqs2...${NC}"
conda install -c bioconda mmseqs2 -y || {
    echo -e "${YELLOW}Trying direct binary...${NC}"
    wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mmseqs.tar.gz
    tar -xzf /tmp/mmseqs.tar.gz -C /tmp/
    sudo cp /tmp/mmseqs/bin/mmseqs /usr/local/bin/
    echo -e "${GREEN}MMseqs2 installed to /usr/local/bin/mmseqs${NC}"
}

# ── 6. Install US-align (TM-score computation) ──────────────────────
echo -e "\n${CYAN}[6/7] Installing US-align...${NC}"
USALIGN_DIR="$HOME/tools/usalign"
mkdir -p "$USALIGN_DIR"
if [ ! -f "$USALIGN_DIR/USalign" ]; then
    wget -q https://zhanggroup.org/US-align/bin/module/USalign.cpp -O "$USALIGN_DIR/USalign.cpp"
    g++ -O3 -o "$USALIGN_DIR/USalign" "$USALIGN_DIR/USalign.cpp" -static 2>/dev/null || \
    g++ -O3 -o "$USALIGN_DIR/USalign" "$USALIGN_DIR/USalign.cpp"
    sudo cp "$USALIGN_DIR/USalign" /usr/local/bin/
    echo -e "${GREEN}US-align compiled and installed${NC}"
else
    echo "  US-align already installed"
fi

# ── 7. Install Python packages ───────────────────────────────────────
echo -e "\n${CYAN}[7/7] Installing Python packages...${NC}"
pip install -r requirements.txt

# ── Protenix (AlphaFold3 reproduction) ──────────────────────────────
echo -e "\n${CYAN}[+] Installing Protenix...${NC}"
if [ ! -d "external/protenix" ]; then
    mkdir -p external
    git clone --depth 1 https://github.com/bytedance/protenix external/protenix
    pip install -e external/protenix/ || echo -e "${YELLOW}Protenix install failed. Check external/protenix/README.md${NC}"
else
    echo "  Protenix already cloned"
fi

# ── Create output directories ────────────────────────────────────────
echo -e "\n${CYAN}[+] Creating output directories...${NC}"
mkdir -p outputs data/raw data/msa data/templates data/pdb_cache data/rfam models/ribonanzanet2 models/protenix

# ── Summary ──────────────────────────────────────────────────────────
echo -e "\n${GREEN}=== Setup complete! ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Download models:         bash scripts/download_models.sh"
echo "  2. Download Rfam database:  bash scripts/download_rfam.sh"
echo "  3. Build PDB cache (1x):    bash scripts/build_pdb_cache.sh"
echo "  4. Place test data:         cp test_sequences.csv data/raw/"
echo "  5. Run pipeline:            python src/pipeline.py"
echo ""
echo -e "Activate environment with: ${CYAN}conda activate rna_folding${NC}"
