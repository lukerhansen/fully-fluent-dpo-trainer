#!/bin/bash
# Run complete DPO training pipeline

set -e  # Exit on error

echo "======================================"
echo "Fully Fluent DPO Training Pipeline"
echo "======================================"
echo ""

# 1. Setup Checks
if [ ! -f "data/prompts.json" ]; then
    echo "❌ ERROR: data/prompts.json not found!"
    echo "Please create this file with your conversation contexts."
    exit 1
fi

# Check if reward model exists in the sibling directory or local folder
RM_PATH="../fully-fluent-reward-model/models/reward_model_final"
LOCAL_RM_PATH="models/reward"

if [ -d "$LOCAL_RM_PATH" ]; then
    RM_PATH="$LOCAL_RM_PATH"
    echo "✓ Found local reward model at $RM_PATH"
elif [ -d "$RM_PATH" ]; then
    echo "✓ Found sibling reward model at $RM_PATH"
else
    echo "❌ ERROR: Reward model not found."
    echo "Please copy it to 'models/reward' or ensure the reward model repo is in '../fully-fluent-reward-model'"
    exit 1
fi

# 2. Generate Candidates
echo ""
echo "======================================"
echo "Step 1/3: Generating Candidates (Self-Play)"
echo "======================================"
python scripts/01_generate_candidates.py \
    --prompts_file "data/prompts.json" \
    --output_file "data/processed/candidates.json"

# 3. Label Data
echo ""
echo "======================================"
echo "Step 2/3: Labeling with Reward Model"
echo "======================================"
python scripts/02_label_preferences.py \
    --reward_model_path "$RM_PATH" \
    --input_file "data/processed/candidates.json" \
    --output_file "data/processed/dpo_pairs.json"

# 4. Train DPO
echo ""
echo "======================================"
echo "Step 3/3: DPO Training"
echo "======================================"
python scripts/03_train_dpo.py \
    --config "config/dpo_config.yaml" \
    --dataset "data/processed/dpo_pairs.json"

echo ""
echo "======================================"
echo "✓ Pipeline Complete!"
echo "======================================"
echo "Final Adapter Saved: models/dpo_final/"
