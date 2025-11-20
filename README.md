# Fully Fluent DPO Trainer

A pipeline for **Direct Preference Optimization (DPO)**.
It uses a trained Reward Model (Judge) to generate synthetic preference data (Chosen vs Rejected) and fine-tunes a base LLM to align with those preferences.

##  Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # Add HF_TOKEN
   ```

2. **Prepare Data**
   Add your prompts to `data/prompts.json`.

3. **Link Reward Model**
   Ensure your `fully-fluent-reward-model` repo is in the parent directory, OR copy the model folder to `models/reward`.

4. **Run Pipeline**
   ```bash
   bash scripts/run_dpo_pipeline.sh
   ```

##  How it Works

1. The base model generates 4 responses for every prompt.
2. The Reward Model scores them. The winner is "Chosen", loser is "Rejected".
3. The model is fine-tuned to maximize the margin between chosen and rejected responses with DPO training.

##  Repository Structure

```text
fully-fluent-dpo-trainer/
├── config/
│   ├── dpo_config.yaml          # Training hyperparameters
│   └── generation_config.yaml   # Settings for generating candidate pairs
├── data/
│   ├── prompts.json             # Your input prompts (conversation history)
│   └── processed/               # Generated candidates and DPO pairs
├── models/
│   └── reward/                  # Copy your trained reward model here
├── outputs/
│   ├── logs/
│   └── checkpoints/
├── scripts/
│   ├── 01_generate_candidates.py
│   ├── 02_label_preferences.py
│   ├── 03_train_dpo.py
│   └── run_dpo_pipeline.sh
├── src/
│   ├── __init__.py
│   └── utils.py                 # Helper functions
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Configuration

### DPO Training (`config/dpo_config.yaml`)
- Uses 4-bit quantization and LoRA for efficiency
- Low learning rate (5e-6) for stable fine-tuning

### Generation (`config/generation_config.yaml`)
- High temperature (0.8) for diverse responses
- Generates 4 candidates per prompt for preference labeling

## Data Format

Your `data/prompts.json` should follow this format:

```json
[
  {
    "context": "Full conversation history...",
    "student_message": "Can you explain the difference between Ser and Estar?"
  },
  {
    "context": "Full conversation history...",
    "student_message": "I made a mistake in the last sentence."
  }
]
```

## Pipeline Steps

1. **Generate Candidates** (`01_generate_candidates.py`)
   - Loads base model
   - Generates 2 diverse responses per prompt
   - Saves to `data/processed/candidates.json`

2. **Label Preferences** (`02_label_preferences.py`)
   - Loads your trained reward model
   - Scores all candidate pairs
   - Creates preference dataset with chosen/rejected pairs
   - Filters out ambiguous pairs (margin < 0.05)

3. **Train DPO** (`03_train_dpo.py`)
   - Fine-tunes base model with DPO loss
   - Uses LoRA for parameter-efficient training
   - Saves adapter to `models/dpo_final/`


##  Tracking

The pipeline supports Weights & Biases integration. Set your API key in `.env` to enable experiment tracking.
