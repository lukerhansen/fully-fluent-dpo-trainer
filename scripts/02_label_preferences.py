import json
import argparse
import sys
import os
from tqdm import tqdm

# Add the sibling repo to python path to import the scorer
# Assuming directories are:
# /workspace/fully-fluent-reward-model
# /workspace/fully-fluent-dpo-trainer
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../fully-fluent-reward-model")
    )
)

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
    parser.add_argument("--margin_threshold", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading Reward Model from {args.reward_model_path}...")
    scorer = RewardModelScorer(args.reward_model_path)
    print("✓ Reward model loaded")

    print(f"Reading candidates from {args.input_file}...")
    with open(args.input_file, "r") as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} candidate sets")

    dpo_dataset = []
    skipped = 0

    print("Labeling preferences...")
    for idx, item in enumerate(tqdm(data)):
        prompt_text = item["original_context"]  # Raw text for reward model
        candidates = item["candidates"]

        # Need at least 2 candidates to form a preference
        if len(candidates) < 2:
            skipped += 1
            continue

        # Score all candidates
        scores = scorer.score_batch(
            contexts=[prompt_text] * len(candidates),
            responses=candidates,
        )

        # Pair scores with candidates and sort descending by score
        score_pairs = sorted(
            [(float(s), c) for s, c in zip(scores, candidates)],
            key=lambda x: x[0],
            reverse=True,
        )

        best_score, best_cand = score_pairs[0]
        worst_score, worst_cand = score_pairs[-1]
        margin = best_score - worst_score

        # Keep only if there is a meaningful difference
        if margin > args.margin_threshold:
            dpo_dataset.append({
                "prompt": item["prompt"],  # Chat-template formatted prompt for DPO
                "chosen": best_cand,
                "rejected": worst_cand,
                "margin": float(margin),
            })
        else:
            skipped += 1

        if (idx + 1) % 10 == 0:
            print(f"  ↪ Scored {idx + 1} / {len(data)} prompts")

    print(f"✓ Created {len(dpo_dataset)} DPO pairs (Skipped {skipped} ambiguous cases)")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w") as f:
        json.dump(dpo_dataset, f, indent=2)
    print(f"✓ Saved labeled pairs to {args.output_file}")


if __name__ == "__main__":
    main()
