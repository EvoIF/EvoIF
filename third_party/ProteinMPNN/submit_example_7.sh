#!/bin/bash
DEFAULT_PDB_DIR="dataset/pdb"
DEFAULT_OUTPUT_DIR="dataset/acr_proteinmpnn"

usage() {
    echo "Usage: $0 [-i <pdb_dir>] [-o <output_dir>] [-h]"
    echo "  -i <pdb_dir>     Directory containing input PDB files (default: $DEFAULT_PDB_DIR)"
    echo "  -o <output_dir>  Directory for ProteinMPNN results (default: $DEFAULT_OUTPUT_DIR)"
    echo "  -h               Show this help message"
    exit 1
}
while getopts "i:o:h" opt; do
    case "$opt" in
        i) PDB_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done
: ${PDB_DIR:=$DEFAULT_PDB_DIR}
: ${OUTPUT_DIR:=$DEFAULT_OUTPUT_DIR}
mkdir -p "$OUTPUT_DIR"

# Path for the intermediate parsed JSONL file
JSONL_PATH="$OUTPUT_DIR/parsed_pdbs.jsonl"

echo "[INFO] Starting PDB parsing: $(date)"
python "third_party/ProteinMPNN/parse_multiple_chains.py" \
    --input_path="$PDB_DIR" \
    --output_path="$JSONL_PATH" \
    --workers 64

echo "[INFO] PDB parsing completed: $(date)"

echo "[INFO] Running ProteinMPNN inference: $(date)"
python "third_party/ProteinMPNN/protein_mpnn_run.py" \
    --jsonl_path "$JSONL_PATH" \
    --out_folder "$OUTPUT_DIR" \
    --num_seq_per_target 1 \
    --sampling_temp "0.1" \
    --unconditional_probs_only 1 \
    --seed 37 \
    --batch_size 1

echo "[SUCCESS] Workflow completed at: $(date)"