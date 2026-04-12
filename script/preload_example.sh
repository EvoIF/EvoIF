#!/bin/bash

# Example script for preloading the CATH dataset

echo "Starting CATH dataset preloading..."

CATH_INPUT_PATH="data/cath/processed_dompdb"                  # CATH raw data path
PRELOADED_OUTPUT_PATH="data/cath/processed_cath" # Output path for preloaded data
STRUC_PROFILE_PATH="/data/cath/foldseek/processed_struc"                      # Structure profile path
IF_PROFILE_PATH="data/cath/processed_ifprobs"                          # Inverse folding profile path

# Check if the input directory exists
if [ ! -d "$CATH_INPUT_PATH" ]; then
    echo "Error: CATH input path does not exist: $CATH_INPUT_PATH"
    echo "Please verify the path and try again."
    exit 1
fi

mkdir -p "$PRELOADED_OUTPUT_PATH"
echo "Input path: $CATH_INPUT_PATH"
echo "Output path: $PRELOADED_OUTPUT_PATH"
echo "Structure profile path: $STRUC_PROFILE_PATH"
echo "Inverse folding profile path: $IF_PROFILE_PATH"

# Execute the preloading script
cd /workspace/fitness/S3F_dev/ || exit
python preload_cath_dataset.py \
    --input_path "$CATH_INPUT_PATH" \
    --output_path "$PRELOADED_OUTPUT_PATH" \
    --struc_profile_path "$STRUC_PROFILE_PATH" \
    --if_profile_path "$IF_PROFILE_PATH" \
    --max_length 512 \
    --profile_types if_profile struc_profile \
    --num_workers 16

echo "Preloading completed successfully!"
echo "Preloaded data is saved at: $PRELOADED_OUTPUT_PATH"