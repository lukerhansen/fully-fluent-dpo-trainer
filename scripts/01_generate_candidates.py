import json
import yaml
import torch
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", default="data/prompts.json")
    parser.add_argument("--output_file", default="data/processed/candidates.json")
    parser.add_argument("--config", default="config/generation_config.yaml")
    args = parser.parse_args()

    # Load Config
    print(f"Loading generation config from {args.config}...")
    with open(args.config) as f:
        gen_cfg = yaml.safe_load(f)
    print(f"✓ Temperature={gen_cfg['temperature']} Top-p={gen_cfg['top_p']} Max tokens={gen_cfg['max_new_tokens']}")

    # Load Data
    # Expected format: [{"context": "Student: Hi", "student_message": "..."}]
    print(f"Loading prompts from {args.prompts_file}...")
    with open(args.prompts_file) as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} prompts")

    # Load Model
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"Initializing {model_name} tokenizer/model (this can take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.backends.mps.is_available():
        target_dtype = torch.float16
        device_map = {"": "mps"}
        print("✓ Detected Apple Silicon (MPS); loading model in float16 on MPS")
    elif torch.cuda.is_available():
        target_dtype = torch.bfloat16
        device_map = "auto"
        print("✓ Detected CUDA; loading model in bfloat16 with auto device map")
    else:
        target_dtype = torch.float32
        device_map = {"": "cpu"}
        print("⚠️ No GPU detected; loading model on CPU (this will be very slow)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=target_dtype,
        device_map=device_map
    )
    print("✓ Model ready")

    results = []

    print(f"Generating candidates for {len(data)} prompts...")
    for idx, item in enumerate(tqdm(data)):
        # Construct the chat prompt
        # Assuming 'context' contains the conversation history string
        full_prompt = f"{item['context']}\nStudent: {item['student_message']}\nTutor:"

        messages = [
            {"role": "system", "content": "You are a helpful language tutor."},
            {"role": "user", "content": full_prompt}
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

        # Generate 2 variants
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=gen_cfg['temperature'],
                top_p=gen_cfg['top_p'],
                max_new_tokens=gen_cfg['max_new_tokens'],
                do_sample=True,
                num_return_sequences=2,
                pad_token_id=tokenizer.eos_token_id
            )

        candidates = []
        for output in outputs:
            decoded = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            candidates.append(decoded)

        results.append({
            "prompt": text_input, # DPO trainer needs the formatted prompt
            "original_context": item['context'],
            "candidates": candidates
        })
        if (idx + 1) % 10 == 0:
            print(f"  ↪ Generated {(idx + 1)} / {len(data)} prompts")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved candidates to {args.output_file}")

if __name__ == "__main__":
    main()
