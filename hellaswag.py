"""
HellaSwag Evaluation Script for GPT-2
Based on Andrej Karpathy's "Let's reproduce GPT-2" video lecture

HellaSwag is a benchmark for evaluating commonsense reasoning in language models.
It's the "North Star" metric proving the model develops common sense rather than 
just memorizing text. Paper: https://arxiv.org/abs/1905.07830

=== HOW IT WORKS ===
Each example has:
- A context (shared prefix)
- 4 possible continuations (only 1 is correct)
- A label indicating which continuation is correct (0-3)

We evaluate by:
1. Concatenating context + each option → 4 sequences
2. Computing the average log-likelihood of each option's tokens
3. Picking the option with the LOWEST loss (highest likelihood)
4. Comparing to the ground truth label

=== BENCHMARK RESULTS ===
Random Guessing: 25.0%
OpenAI GPT-2 (124M): 29.4%
Karpathy's Reproduction (124M): ~29.9% (at step 20k-25k, trained on FineWeb-Edu)
GPT-3 Small (124M): 33.7%
Human: ~95.6%

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14%
  (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955
  (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89%
  (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893
  (completion style)

=== WHY IT MATTERS ===
- Unlike validation loss which can jitter, HellaSwag is "well-behaved"
- Accuracy climbs smoothly, providing early signal that training is healthy
- The "crossover point" where we beat OpenAI's 29.4% happens around step 25k
- Proves modern FineWeb-Edu dataset is higher quality than 2019's WebText

=== DATASET ===
Uses the rowan/hellaswag dataset from GitHub.
The validation split has 10,042 examples.
"""

import os
import json
import requests
import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------------------------------------------------------

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

enc = tiktoken.get_encoding("gpt2")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    """Downloads HellaSwag dataset split to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    return data_filename

def render_example(example):
    """
    Given the example as a dictionary, return:
    - tokens (the tokens of context + each of 4 candidates)
    - mask (is 1 in the region where we evaluate likelihood)
    - label (the index of the correct completion)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this example
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # note: prepend space for tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during tokenization because row lengths can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    """Iterate over examples in the split"""
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

def get_most_likely_row(tokens, mask, logits):
    """
    Given tokens, mask, and logits, return the index of the most likely completion.
    Used for inline HellaSwag evaluation during training.
    """
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

@torch.no_grad()
def evaluate(model, device):
    """Evaluate the model on HellaSwag validation set"""
    model.eval()
    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    for example in tqdm(iterate_examples("val"), total=10042, desc="HellaSwag"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits from the model
        with torch.autocast(device_type=device if device != "cuda" else "cuda", dtype=torch.bfloat16):
            logits, _ = model(tokens)

        # evaluate the autoregressive loss at all positions
        shift_logits = logits[:, :-1, :].contiguous()
        shift_tokens = tokens[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        # calculate per-token loss
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_tokens = shift_tokens.view(-1)
        loss_flat = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
        loss = loss_flat.view(shift_tokens.size())

        # mask out the context tokens (only score the ending tokens)
        masked_loss = loss * shift_mask

        # sum loss for each row (each completion option)
        sum_loss = masked_loss.sum(dim=1)  # (4,)
        # count tokens for normalization
        num_tokens = shift_mask.sum(dim=1)  # (4,)
        # average loss (normalized by number of tokens)
        avg_loss = sum_loss / num_tokens  # (4,)

        # pick the completion with the lowest loss
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

    acc = num_correct / num_total
    acc_norm = num_correct_norm / num_total
    print(f"HellaSwag accuracy: {num_correct}/{num_total} = {acc:.4f}")
    print(f"HellaSwag accuracy (normalized): {num_correct_norm}/{num_total} = {acc_norm:.4f}")
    return acc_norm


# -----------------------------------------------------------------------------
# Main evaluation loop (standalone)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="gpt2", help="Model to evaluate: gpt2, gpt2-medium, gpt2-large, gpt2-xl, or path to checkpoint")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use: cuda, cpu, mps")
    args = parser.parse_args()

    # Load model
    if args.model.startswith("gpt2"):
        # Load pretrained GPT-2 from our implementation
        import sys
        sys.path.append(os.path.dirname(__file__))
        from train_gpt2 import GPT
        model = GPT.from_pretrained(args.model)
        model.to(args.device)
    else:
        # Load from checkpoint
        print(f"Loading checkpoint from {args.model}")
        checkpoint = torch.load(args.model, map_location=args.device)
        from train_gpt2 import GPT, GPTConfig
        config = GPTConfig(**checkpoint["config"])
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        model.to(args.device)

    # Evaluate
    evaluate(model, args.device)
