import json
import argparse
import sys
import os
from tqdm import tqdm

# Add the sibling repo to python path to import the scorer
# Assuming directories are:
# /workspace/fully-fluent-reward-model
# /workspace/fully-fluent-dpo-trainer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fully-fluent-reward-model")))

try:
    from src.reward_model.inference import RewardModelScorer
except ImportError:
    print("❌ Error: Could not import RewardModelScorer.")
    print("Make sure the 'fully-fluent-reward-model' repo is located next to this one.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model_path", required=True)
    parser.add_argument("--input_file", default="data/processed/candidates.json")
    parser.add_argument("--output_file", default="data/processed/dpo_pairs.json")
    args = parser.parse_args()

    print(f"Loading Reward Model from {args.reward_model_path}...")
    scorer = RewardModelScorer(args.reward_model_path)
    print("✓ Reward model loaded")

    print(f"Reading candidates from {args.input_file}...")
    with open(args.input_file) as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} candidate pairs")

    dpo_dataset = []
    skipped = 0

    print("Labeling preferences...")
    for idx, item in enumerate(tqdm(data)):
        prompt_text = item['original_context'] # Pass raw text to reward model, not chat template
        cand_a = item['candidates'][0]
        cand_b = item['candidates'][1]

        # Score both
        scores = scorer.score_batch(
            contexts=[prompt_text, prompt_text],
            responses=[cand_a, cand_b]
        )
        score_a, score_b = scores[0], scores[1]

        # Decide winner
        if score_a > score_b:
            chosen, rejected = cand_a, cand_b
            margin = score_a - score_b
        else:
            chosen, rejected = cand_b, cand_a
            margin = score_b - score_a

        # Filter: Keep only if there is a meaningful difference (e.g., > 0.05)
        if margin > 0.05:
            dpo_dataset.append({
                "prompt": item['prompt'], # Use the chat-template formatted prompt for DPO training
                "chosen": chosen,
                "rejected": rejected,
                "margin": float(margin)
            })
        else:
            skipped += 1
        if (idx + 1) % 10 == 0:
            print(f"  ↪ Scored {(idx + 1)} / {len(data)} prompts")

    print(f"✓ Created {len(dpo_dataset)} DPO pairs (Skipped {skipped} ambiguous pairs)")

    with open(args.output_file, 'w') as f:
        json.dump(dpo_dataset, f, indent=2)
    print(f"✓ Saved labeled pairs to {args.output_file}")

if __name__ == "__main__":
    main()
