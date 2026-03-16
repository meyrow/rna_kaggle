#!/usr/bin/env bash
# scripts/run_pipeline.sh — End-to-end pipeline runner
# Usage: bash scripts/run_pipeline.sh [input_csv] [output_csv]

set -euo pipefail
CYAN='\033[0;36m'; GREEN='\033[0;32m'; NC='\033[0m'

INPUT="${1:-data/raw/test_sequences.csv}"
OUTPUT="${2:-outputs/submission.csv}"
CONFIG="config/config.yaml"

echo -e "${CYAN}=== RNA 3D Folding Pipeline ===${NC}"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Config: $CONFIG"

# Activate conda env if not active
if [[ "${CONDA_DEFAULT_ENV:-}" != "rna_folding" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate rna_folding 2>/dev/null || true
fi

# Check GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')" 2>/dev/null || true

# Run
time python3 src/pipeline.py \
    --config "$CONFIG" \
    --input  "$INPUT" \
    --output "$OUTPUT"

# Validate
echo -e "\n${CYAN}Validating submission...${NC}"
python3 - <<EOF
from src.submission import SubmissionBuilder
sb = SubmissionBuilder()
ok = sb.validate("$OUTPUT")
print("Validation:", "PASSED" if ok else "FAILED")
EOF

echo -e "\n${GREEN}Done! Submit: $OUTPUT${NC}"
