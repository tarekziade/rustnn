#!/usr/bin/env python3
"""
HuggingFace Transformers baseline for SmolLM-135M generation.

This script generates text using the official HuggingFace implementation
to establish a baseline for comparison with WebNN execution.
"""

import sys
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"


def main():
    parser = argparse.ArgumentParser(description="SmolLM-135M HuggingFace baseline")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed token-by-token output",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SmolLM-135M HuggingFace Transformers Baseline")
    print("=" * 70)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID)
    model.eval()

    # Tokenize prompt
    print(f"\nPrompt: '{args.prompt}'")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    prompt_tokens = input_ids[0].tolist()
    print(f"Tokens: {prompt_tokens} ({len(prompt_tokens)} tokens)")
    for i, token_id in enumerate(prompt_tokens):
        token_text = tokenizer.decode([token_id])
        print(f"  Token {i}: '{token_text}' (id={token_id})")

    if args.verbose:
        print("\n" + "=" * 70)
        print("Token-by-Token Generation (Verbose Mode)")
        print("=" * 70)

        # Manual token-by-token generation for detailed inspection
        past_key_values = None
        current_ids = input_ids
        current_mask = attention_mask

        generated_tokens = []

        for step in range(args.max_new_tokens):
            print(f"\n--- Step {step + 1} ---")
            print(f"Input IDs shape: {current_ids.shape}")
            print(f"Attention mask shape: {current_mask.shape}")

            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            # Get logits and select next token
            logits = outputs.logits  # [batch, seq_len, vocab_size]
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            next_token_id = torch.argmax(next_token_logits).item()

            token_text = tokenizer.decode([next_token_id])
            generated_tokens.append(next_token_id)

            print(f"Next token: '{token_text}' (id={next_token_id})")
            print(f"Logits shape: {logits.shape}")
            print(f"Top-5 logits: {torch.topk(next_token_logits, 5).indices.tolist()}")

            if args.verbose and step == 0:
                # Print cache shapes on first step
                print(f"KV cache shapes:")
                if outputs.past_key_values:
                    for layer_idx, (key, value) in enumerate(outputs.past_key_values):
                        print(f"  Layer {layer_idx}: key={key.shape}, value={value.shape}")
                        if layer_idx >= 2:  # Just show first few layers
                            print(f"  ... ({len(outputs.past_key_values)} layers total)")
                            break

            # Update for next iteration
            current_ids = torch.tensor([[next_token_id]])  # [1, 1]
            current_mask = torch.cat([current_mask, torch.ones((1, 1), dtype=torch.long)], dim=1)
            past_key_values = outputs.past_key_values

            # Stop if EOS token
            if next_token_id == tokenizer.eos_token_id:
                print("Reached end-of-sequence token")
                break
    else:
        # Standard generation
        print("\nGenerating tokens...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy decoding
                use_cache=True,
            )

        generated_tokens = outputs[0, len(prompt_tokens):].tolist()

    # Print results
    print("\n" + "=" * 70)
    print("Generated Tokens")
    print("=" * 70)

    for i, token_id in enumerate(generated_tokens):
        token_text = tokenizer.decode([token_id])
        print(f"Step {i}: '{token_text}' (id={token_id})")

    print("\n" + "=" * 70)
    print("Full Generated Text")
    print("=" * 70)

    if args.verbose:
        full_ids = prompt_tokens + generated_tokens
    else:
        full_ids = outputs[0].tolist()

    full_text = tokenizer.decode(full_ids)
    print(f"\n{full_text}\n")

    print("=" * 70)
    print(f"Total tokens generated: {len(generated_tokens)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
