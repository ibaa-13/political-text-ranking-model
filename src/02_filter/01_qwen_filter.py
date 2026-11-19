"""Sample texts and classify clear US left-right positions with a HF model."""

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
INPUT_PATHS = [Path.home() / "paragraphs.parquet", PROJECT_ROOT / "data/processed/paragraphs.parquet"]
OUTPUT_PATH = Path.home() / "ideological_qwen_results.csv"
SEED = 20261119
YES_THRESHOLD = 0.95
SYSTEM_PROMPT = (
    "Classify congressional speech excerpts for a study of US political ideology. "
    "Answer YES only if the excerpt itself clearly endorses or opposes a substantive "
    "policy or political principle that can be placed on a US liberal-conservative "
    "left-right scale. A topic being political, support for a technical or bipartisan "
    "bill, or a speaker's party is not enough. Do not infer unstated positions. "
    "If the direction is unclear, mixed, merely descriptive, procedural, ceremonial, "
    "or not confidently placeable on the left-right scale, answer NO."
)


def score_next_token(model, tokenizer, prompt):
    """Compare continuation likelihoods for YES and NO without padded batches."""
    yes_ids = tokenizer.encode(" YES", add_special_tokens=False)
    no_ids = tokenizer.encode(" NO", add_special_tokens=False)
    if not yes_ids or not no_ids:
        raise ValueError("The tokenizer returned an empty YES or NO label.")
    input_ids = torch.tensor([prompt], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        yes_score = no_score = 0.0
        for label_ids, label in ((yes_ids, "YES"), (no_ids, "NO")):
            current_ids, current_mask = input_ids, attention_mask
            total = 0.0
            for token_id in label_ids:
                logits = model(input_ids=current_ids, attention_mask=current_mask).logits[0, -1].float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError(
                        f"The model returned non-finite logits while scoring {label}. "
                        "This points to a model/precision issue, not a label-tokenization issue."
                    )
                total += float(logits.log_softmax(dim=-1)[token_id].item())
                next_id = torch.tensor([[token_id]], device=current_ids.device)
                current_ids = torch.cat((current_ids, next_id), dim=1)
                current_mask = torch.cat((current_mask, torch.ones_like(next_id)), dim=1)
            if label == "YES":
                yes_score = total
            else:
                no_score = total
    return yes_score, no_score


def load_model(model_name, requested_device):
    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable. Use --device cpu or run on a GPU node.")
    print(f"Loading {model_name} on {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cpu")
    return tokenizer, model.eval(), device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-14B", help="Hugging Face model ID")
    parser.add_argument("--n-texts", type=int, default=300, help="number of random texts to classify")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.n_texts < 1:
        parser.error("--n-texts must be positive")

    input_path = next((p for p in INPUT_PATHS if p.exists()), None)
    if input_path is None:
        raise SystemExit("Could not find paragraphs.parquet in ~/ or the project's data/processed folder.")
    data = pd.read_parquet(input_path)
    required = {"para_id", "text"}
    if not required.issubset(data.columns):
        raise ValueError(f"Input is missing required columns: {sorted(required - set(data.columns))}")
    data = data.dropna(subset=["para_id", "text"])
    if len(data) < args.n_texts:
        raise ValueError(f"Requested {args.n_texts} texts but only {len(data)} are available")
    data = data.sample(n=args.n_texts, random_state=SEED).reset_index(drop=True)
    tokenizer, model, device = load_model(args.model, args.device)

    labels, yes_scores, no_scores, yes_confidence = [], [], [], []
    for i, text in enumerate(data["text"].astype(str), start=1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Does this excerpt clearly express a position placeable on the US left-right ideological scale? Answer with exactly YES or NO.\n\nExcerpt:\n{text}\n\nLabel:"},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=False,
        )
        token_ids = encoded["input_ids"]
        if torch.is_tensor(token_ids):
            token_ids = token_ids.squeeze(0).tolist()
        elif token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        yes, no = score_next_token(model, tokenizer, list(token_ids))
        if not math.isfinite(yes) or not math.isfinite(no):
            raise RuntimeError(f"Non-finite YES/NO score at excerpt {i}; stopping.")
        a, b = math.exp(yes - max(yes, no)), math.exp(no - max(yes, no))
        p_yes = a / (a + b)
        labels.append("YES" if p_yes >= YES_THRESHOLD else "NO")
        yes_scores.append(yes)
        no_scores.append(no)
        yes_confidence.append(p_yes)
        if i % 10 == 0 or i == len(data):
            print(f"Classified {i}/{len(data)}", flush=True)

    data["qwen_label"] = labels
    data["qwen_yes_logprob"] = yes_scores
    data["qwen_no_logprob"] = no_scores
    data["qwen_yes_confidence"] = yes_confidence
    data["qwen_model"] = args.model
    data["qwen_device"] = device
    data["qwen_prompt_version"] = "clear-left-right-v2-95pct"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data)} rows to {OUTPUT_PATH}; {data['qwen_label'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
