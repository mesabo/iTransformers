#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodelist=ai-gpgpu14

# Load necessary environment
source ~/.bashrc
hostname

# Check for GPU/MPS/CPU availability
if [[ -z "$CUDA_VISIBLE_DEVICES" ]] || [[ $(nvidia-smi | grep -c "GPU") -eq 0 ]]; then
    echo "No GPU devices detected. Defaulting to CPU or MPS if available."
    DEVICE="cpu"
    if [[ "$(uname -s)" == "Darwin" ]] && command -v system_profiler &>/dev/null && system_profiler SPHardwareDataType | grep -q "Apple M"; then
        DEVICE="mps" # Apple Silicon MPS support
        echo "MPS device detected. Running on MPS."
    else
        echo "CPU device selected."
    fi
else
    echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
    DEVICE="cuda"
fi

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate Python environment
ENV_NAME="itransformers"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    if ! command -v conda &>/dev/null; then
        echo "Conda command not found. Ensure Miniconda/Anaconda is installed and added to PATH."
        exit 1
    fi
    source activate "$ENV_NAME" || { echo "Failed to activate conda environment '$ENV_NAME'"; exit 1; }
fi

# Default configurations
DATASETS=("Iris")  # Replace with lightweight datasets for iTransformers
MODELS=("transformer")
BATCH_SIZES=("32")
NUM_CLASSES="10"
LEARNING_RATE="0.001"
EPOCHS=("10")
OUTPUT_DIR="output"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR" || { echo "Error: Unable to create directory $OUTPUT_DIR"; exit 1; }

# Loop through configurations
for EPOCH in "${EPOCHS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
          OUTPUT_FILE="$OUTPUT_DIR/${MODEL}_${DATASET}_batch${BATCH_SIZE}_epoch${EPOCH}.log"

          echo "Running MODEL=$MODEL, DATASET=$DATASET, BATCH_SIZE=$BATCH_SIZE, EPOCHS=$EPOCH"

          # Execute the Python script with the current configuration
          python -u main.py \
              --task time_series \
              --model "$MODEL" \
              --data_path "./data/$DATASET" \
              --save_path "$OUTPUT_DIR" \
              --batch_size "$BATCH_SIZE" \
              --epochs "$EPOCH" \
              --learning_rate "$LEARNING_RATE" \
              --device "$DEVICE" > "$OUTPUT_FILE" 2>&1

          if [[ $? -ne 0 ]]; then
              echo "Error: Python script failed for MODEL=$MODEL, DATASET=$DATASET"
              exit 1
          fi

          echo "Execution complete for MODEL=$MODEL, DATASET=$DATASET, BATCH_SIZE=$BATCH_SIZE, EPOCHS=$EPOCH. Output logged in $OUTPUT_FILE."
      done
    done
  done
done

echo "All executions complete."