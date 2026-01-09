#!/usr/bin/env python3
"""
Test the original ONNX model with ONNX Runtime directly.
This verifies if the model is correct before any WebNN conversion.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig

HF_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
MAX_CACHE_SIZE = 128


def main():
    parser = argparse.ArgumentParser(description="Test original ONNX model")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default=str(Path.home() / "Dev" / "SmolLM-135-webnn" / "smollm-135.onnx"),
        help="Path to ONNX model file",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Original ONNX Model Test (ONNX Runtime)")
    print("=" * 70)

    # Load tokenizer and config
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    config = AutoConfig.from_pretrained(HF_MODEL_ID)

    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", 3)
    hidden_size = config.hidden_size
    num_attention_heads = getattr(config, "num_attention_heads", 9)
    head_dim = hidden_size // num_attention_heads

    print(f"\nModel Configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")

    # Tokenize prompt
    print(f"\nPrompt: '{args.prompt}'")
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="np")
    prompt_tokens = prompt_ids[0].tolist()
    prompt_len = len(prompt_tokens)
    print(f"Tokens: {prompt_tokens} ({prompt_len} tokens)")

    # Load ONNX model
    print(f"\nLoading ONNX model: {args.onnx_model}")
    session = ort.InferenceSession(args.onnx_model)

    # Get input names
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"Model has {len(input_names)} inputs")
    print(f"  First 5 inputs: {input_names[:5]}")

    # Check if model has past_key_values inputs
    # Original: "past_key_values.0.key", Converted: "past_key_values_0_key"
    kv_input_names = [name for name in input_names if "past_key_values" in name and name != "past_key_values"]
    has_kv_inputs = len(kv_input_names) > 0

    if has_kv_inputs:
        # Check shape of first KV input to determine past_sequence_length
        first_kv_input = [inp for inp in session.get_inputs() if inp.name == kv_input_names[0]][0]
        kv_shape = first_kv_input.shape
        print(f"  KV input shape: {kv_shape}")

        # Extract past_sequence_length from shape (typically [batch, heads, past_seq_len, head_dim])
        if isinstance(kv_shape[2], int):
            past_seq_len = kv_shape[2]
        else:
            # Dynamic dimension, use 1 as default
            past_seq_len = 1
        print(f"  past_sequence_length: {past_seq_len}")
    else:
        past_seq_len = 0
        print(f"  No KV inputs (past_sequence_length = 0)")

    # Prepare inputs for prompt processing
    input_ids = np.array([prompt_tokens], dtype=np.int64)

    # Attention mask size depends on past_sequence_length
    attention_mask_size = prompt_len + past_seq_len
    attention_mask = np.zeros((1, attention_mask_size), dtype=np.int64)
    attention_mask[0, past_seq_len:past_seq_len + prompt_len] = 1

    position_ids = np.array([list(range(prompt_len))], dtype=np.int64)

    # Detect naming pattern (dot vs underscore) for KV inputs
    uses_dots = False
    if has_kv_inputs and len(kv_input_names) > 0:
        uses_dots = "." in kv_input_names[0]

    # Prepare KV cache inputs
    feed_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    if has_kv_inputs:
        print(f"  Adding {len(kv_input_names)} KV cache inputs (shape: [1, {num_kv_heads}, {past_seq_len}, {head_dim}])")

        for layer in range(num_layers):
            if uses_dots:
                key_name = f"past_key_values.{layer}.key"
                value_name = f"past_key_values.{layer}.value"
            else:
                key_name = f"past_key_values_{layer}_key"
                value_name = f"past_key_values_{layer}_value"

            feed_dict[key_name] = np.zeros(
                (1, num_kv_heads, past_seq_len, head_dim), dtype=np.float32
            )
            feed_dict[value_name] = np.zeros(
                (1, num_kv_heads, past_seq_len, head_dim), dtype=np.float32
            )

    print(f"\nProcessing prompt...")

    # Run inference
    outputs = session.run(None, feed_dict)

    # Get logits
    logits = outputs[0]  # First output is logits
    print(f"Logits shape: {logits.shape}")

    # Get first generated token
    next_token_id = int(np.argmax(logits[0, -1, :]))
    first_token_text = tokenizer.decode([next_token_id])

    print(f"First generated token: '{first_token_text}' (id={next_token_id})")
    print(f"Top 5 tokens:")
    top5_indices = np.argsort(logits[0, -1, :])[-5:][::-1]
    for idx in top5_indices:
        token_text = tokenizer.decode([idx])
        prob = logits[0, -1, idx]
        print(f"  {idx}: '{token_text}' (logit={prob:.2f})")

    # Extract KV cache for generation
    kv_cache = {}
    if has_kv_inputs:
        for layer in range(num_layers):
            # ONNX model outputs are: logits, present_0_key, present_0_value, present_1_key, ...
            key_idx = 1 + layer * 2
            value_idx = 2 + layer * 2

            if uses_dots:
                key_name = f"past_key_values.{layer}.key"
                value_name = f"past_key_values.{layer}.value"
            else:
                key_name = f"past_key_values_{layer}_key"
                value_name = f"past_key_values_{layer}_value"

            kv_cache[key_name] = outputs[key_idx]
            kv_cache[value_name] = outputs[value_idx]

    # Generate tokens
    generated_tokens = [next_token_id]
    current_pos = prompt_len

    print(f"\nGenerating {args.max_new_tokens - 1} more tokens...")

    for step in range(args.max_new_tokens - 1):
        # Prepare inputs for next token
        feed_dict = {
            "input_ids": np.array([[next_token_id]], dtype=np.int64),
            "attention_mask": np.ones((1, current_pos + past_seq_len + 1), dtype=np.int64),
            "position_ids": np.array([[current_pos]], dtype=np.int64),
        }

        if has_kv_inputs:
            feed_dict.update(kv_cache)

        # Run inference
        outputs = session.run(None, feed_dict)

        # Get next token
        logits = outputs[0]
        next_token_id = int(np.argmax(logits[0, 0, :]))
        token_text = tokenizer.decode([next_token_id])
        generated_tokens.append(next_token_id)

        print(f"  Step {step + 1}: '{token_text}' (id={next_token_id})")

        # Update KV cache
        if has_kv_inputs:
            for layer in range(num_layers):
                key_idx = 1 + layer * 2
                value_idx = 2 + layer * 2
                kv_cache[f"past_key_values.{layer}.key"] = outputs[key_idx]
                kv_cache[f"past_key_values.{layer}.value"] = outputs[value_idx]

        current_pos += 1

        # Stop if EOS
        if next_token_id == tokenizer.eos_token_id:
            print("\nReached EOS token")
            break

    # Print results
    print("\n" + "=" * 70)
    print("Generated Text")
    print("=" * 70)

    full_tokens = prompt_tokens + generated_tokens
    full_text = tokenizer.decode(full_tokens)
    print(f"\n{full_text}\n")

    print("=" * 70)
    print("Statistics")
    print("=" * 70)
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print(f"Generated tokens: {len(generated_tokens)}")
    print(f"Total tokens: {len(full_tokens)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
