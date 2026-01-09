#!/usr/bin/env python3
"""
SmolLM-135M Token-by-Token Generation with Pre-Allocated KV Cache.

This script demonstrates true token-by-token generation with WebNN's
static shape requirement by pre-allocating the full KV cache and using
attention masking.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
import webnn

HF_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"


def main():
    parser = argparse.ArgumentParser(description="SmolLM-135M with pre-allocated cache")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "Dev" / "SmolLM-135-webnn" / "smollm-135-cached512.webnn"),
        help="Path to WebNN model file",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(Path.home() / "Dev" / "SmolLM-135-webnn" / "smollm-135.manifest.json"),
        help="Path to manifest file",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=str(Path.home() / "Dev" / "SmolLM-135-webnn" / "smollm-135.weights"),
        help="Path to weights file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=512,
        help="Pre-allocated cache size (must match model conversion)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device type (cpu, gpu)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SmolLM-135M Pre-Allocated Cache Token-by-Token Generation")
    print("=" * 70)

    # Load tokenizer and config
    print("\nLoading tokenizer and config...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    config = AutoConfig.from_pretrained(HF_MODEL_ID)

    num_layers = config.num_hidden_layers
    num_key_value_heads = getattr(config, "num_key_value_heads", 3)
    hidden_size = config.hidden_size
    num_attention_heads = getattr(config, "num_attention_heads", 9)
    head_dim = hidden_size // num_attention_heads

    print(f"\nModel Configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  KV heads: {num_key_value_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Max context: {args.max_context_length} tokens")

    # Calculate memory usage
    kv_memory_mb = (
        num_layers * 2 * 4 * 1 * num_key_value_heads * args.max_context_length * head_dim
    ) / (1024 * 1024)
    print(f"  KV cache memory: {kv_memory_mb:.1f} MB")

    # Tokenize prompt
    print(f"\nPrompt: '{args.prompt}'")
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="np")
    prompt_tokens = prompt_ids[0].tolist()
    print(f"Tokens: {prompt_tokens} ({len(prompt_tokens)} tokens)")

    if len(prompt_tokens) > args.max_context_length:
        print(f"\nERROR: Prompt exceeds max context length!")
        return 1

    # Load WebNN model
    print(f"\nLoading WebNN model...")
    print(f"  Model: {args.model_path}")
    print(f"  Manifest: {args.manifest_path}")
    print(f"  Weights: {args.weights_path}")

    graph = webnn.MLGraph.load(
        args.model_path,
        manifest_path=args.manifest_path,
        weights_path=args.weights_path,
    )
    print("Model loaded successfully!")

    # Create context
    print(f"\nCreating WebNN context (device={args.device})...")
    ml = webnn.ML()
    context = ml.create_context(device_type=args.device)
    print("Context created successfully!")

    # Pre-allocate full KV cache
    print(f"\nPre-allocating KV cache: [1, {num_key_value_heads}, {args.max_context_length}, {head_dim}]")
    kv_cache = {}
    for layer in range(num_layers):
        kv_cache[f"past_key_values_{layer}_key"] = np.zeros(
            (1, num_key_value_heads, args.max_context_length, head_dim),
            dtype=np.float32
        )
        kv_cache[f"past_key_values_{layer}_value"] = np.zeros(
            (1, num_key_value_heads, args.max_context_length, head_dim),
            dtype=np.float32
        )

    # Initialize attention mask
    print(f"Initializing attention mask: [1, {args.max_context_length + 1}]")
    attention_mask = np.zeros((1, args.max_context_length + 1), dtype=np.int64)

    # Generate tokens
    print("\n" + "=" * 70)
    print("Token-by-Token Generation")
    print("=" * 70)

    current_pos = 0
    generated_tokens = []
    full_text = args.prompt

    # Process all prompt tokens one by one
    # Note: This processes each token seeing only previous tokens via cache
    print(f"\nProcessing prompt ({len(prompt_tokens)} tokens) token-by-token...")
    for i, token_id in enumerate(prompt_tokens):
        token_text = tokenizer.decode([token_id])
        print(f"  Token {i}: '{token_text}' (id={token_id})", end="", flush=True)

        # Enable attention for current position
        attention_mask[0, current_pos] = 1

        # Build feed dict
        feed = {
            "input_ids": np.array([[token_id]], dtype=np.int64),
            "attention_mask": attention_mask,
            "position_ids": np.array([[current_pos]], dtype=np.int64),
            **kv_cache
        }

        # Execute model
        result = context.compute(graph, feed)

        # Update cache at current position
        for layer in range(num_layers):
            present_key = result[f"present_{layer}_key"]  # [1, 3, current_pos+1, 64]
            present_value = result[f"present_{layer}_value"]

            # Extract just the new position (last position in sequence dimension)
            new_key = present_key[:, :, current_pos:current_pos+1, :]
            new_value = present_value[:, :, current_pos:current_pos+1, :]

            kv_cache[f"past_key_values_{layer}_key"][:, :, current_pos:current_pos+1, :] = new_key
            kv_cache[f"past_key_values_{layer}_value"][:, :, current_pos:current_pos+1, :] = new_value

        current_pos += 1
        print(" ✓")

    print(f"Prompt processed, cache filled up to position {current_pos}")

    # Generate new tokens
    print(f"\nGenerating {args.max_new_tokens} new tokens...")

    # Get first token from the last prompt token's logits
    logits = result["logits"]  # [1, 1, vocab_size]
    next_token_id = int(np.argmax(logits[0, 0, :]))
    print(f"First generated token: '{tokenizer.decode([next_token_id])}' (id={next_token_id})")

    for step in range(args.max_new_tokens):
        if current_pos >= args.max_context_length:
            print(f"\nReached context limit ({args.max_context_length} tokens)")
            break

        # Enable attention for current position
        attention_mask[0, current_pos] = 1

        # Build feed dict
        feed = {
            "input_ids": np.array([[next_token_id]], dtype=np.int64),
            "attention_mask": attention_mask,
            "position_ids": np.array([[current_pos]], dtype=np.int64),
            **kv_cache
        }

        # Execute model
        result = context.compute(graph, feed)

        # Update cache at current position
        for layer in range(num_layers):
            present_key = result[f"present_{layer}_key"]  # [1, 3, current_pos+1, 64]
            present_value = result[f"present_{layer}_value"]

            # Extract just the new position (last position in sequence dimension)
            new_key = present_key[:, :, current_pos:current_pos+1, :]
            new_value = present_value[:, :, current_pos:current_pos+1, :]

            kv_cache[f"past_key_values_{layer}_key"][:, :, current_pos:current_pos+1, :] = new_key
            kv_cache[f"past_key_values_{layer}_value"][:, :, current_pos:current_pos+1, :] = new_value

        # Get next token from logits
        logits = result["logits"]  # [1, 1, vocab_size]
        next_token_id = int(np.argmax(logits[0, 0, :]))

        token_text = tokenizer.decode([next_token_id])
        generated_tokens.append(next_token_id)
        full_text += token_text

        print(f"Step {len(prompt_tokens) + step}: '{token_text}' (id={next_token_id})")

        current_pos += 1

        # Stop if we hit end-of-sequence token
        if next_token_id == tokenizer.eos_token_id:
            print("Reached end-of-sequence token")
            break

    # Print final output
    print("\n" + "=" * 70)
    print("Generated Text")
    print("=" * 70)
    print(f"\n{full_text}\n")

    print("=" * 70)
    print(f"Total tokens generated: {len(generated_tokens)}")
    print(f"Context used: {current_pos}/{args.max_context_length} tokens")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
