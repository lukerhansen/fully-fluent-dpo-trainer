import yaml
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dpo_config.yaml")
    parser.add_argument("--dataset", default="data/processed/dpo_pairs.json")
    args = parser.parse_args()

    print(f"Loading training config from {args.config}...")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1. Quantization Config
    print("Preparing BitsAndBytes config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. Load Base Model
    print(f"Loading base model: {cfg['model']['base_model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['base_model_name'],
        quantization_config=bnb_config if cfg['model']['load_in_4bit'] else None,
        device_map="auto",
        attn_implementation="flash_attention_2" if cfg['model']['use_flash_attention'] else "eager"
    )
    print("✓ Base model ready")

    # Load Tokenizer (Padding side must be left for generation/DPO usually)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['base_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer ready")

    # 3. Load Dataset
    print(f"Loading DPO dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"✓ Dataset size: {len(dataset)} examples")

    # 4. LoRA Config
    print("Configuring LoRA adapters...")
    peft_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        lora_dropout=cfg['lora']['dropout'],
        target_modules=cfg['lora']['target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. DPO Config
    print("Building DPO training arguments...")
    dpo_args = DPOConfig(
        output_dir=cfg['training']['output_dir'],
        beta=cfg['training']['beta'],
        learning_rate=float(cfg['training']['learning_rate']),
        per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        num_train_epochs=cfg['training']['num_train_epochs'],
        warmup_ratio=cfg['training']['warmup_ratio'],
        logging_steps=cfg['training']['logging_steps'],
        save_steps=cfg['training']['save_steps'],
        fp16=False,
        bf16=True, # Recommended for Ampere GPUs (A10, A100, 3090)
        max_length=cfg['training']['max_length'],
        max_prompt_length=cfg['training']['max_prompt_length'],
        remove_unused_columns=False,
        report_to="wandb"
    )

    # 6. Initialize Trainer
    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # TRL will automagically load the ref model in 4bit if None is passed with PEFT
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training loop...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model("models/dpo_final")
    print("✓ Training complete! Adapters written to models/dpo_final")

if __name__ == "__main__":
    main()
