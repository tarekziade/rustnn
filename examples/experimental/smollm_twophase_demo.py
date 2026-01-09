#!/usr/bin/env python3
"""
Two-phase SmolLM generation with correct output quality.

PHASE 1: Process entire prompt as batch (establishes correct KV cache)
PHASE 2: Token-by-token generation using pre-filled cache

This approach produces outputs matching HuggingFace Transformers.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
import webnn

HF_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
MAX_CACHE_SIZE = 128


def main():
    parser = argparse.ArgumentParser(description="SmolLM two-phase generation")
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
        "--model-dir",
        type=str,
        default=str(Path.home() / "Dev" / "SmolLM-135-webnn"),
        help="Directory containing WebNN models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device type (cpu, gpu)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed execution info",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SmolLM Two-Phase Generation")
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
    print(f"  Max cache size: {MAX_CACHE_SIZE} tokens")

    # Tokenize prompt
    print(f"\nPrompt: '{args.prompt}'")
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="np")
    prompt_tokens = prompt_ids[0].tolist()
    prompt_len = len(prompt_tokens)

    print(f"Tokens: {prompt_tokens} ({prompt_len} tokens)")

    if prompt_len > MAX_CACHE_SIZE:
        print(f"\nERROR: Prompt too long ({prompt_len} > {MAX_CACHE_SIZE})")
        return 1

    # Create WebNN context
    print(f"\nCreating WebNN context (device={args.device})...")
    ml = webnn.ML()
    context = ml.create_context(device_type=args.device)
    print("Context created!")

    # Model paths
    model_dir = Path(args.model_dir)
    # Try clean prompt model first (past_sequence_length=0), fall back to workaround model
    prompt_model_clean = model_dir / "smollm-135-prompt-clean.webnn"
    if prompt_model_clean.exists():
        prompt_model_path = prompt_model_clean
        print(f"Using clean prompt model (past_sequence_length=0)")
    else:
        prompt_model_path = model_dir / "smollm-135-prompt.webnn"
        print(f"Using workaround prompt model (past_sequence_length=1)")

    gen_model_path = model_dir / "smollm-135-generate.webnn"
    manifest_path = model_dir / "smollm-135.manifest.json"
    weights_path = model_dir / "smollm-135.weights"

    # =====================================================
    # PHASE 1: PROMPT PROCESSING (BATCH)
    # =====================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Prompt Processing (Batch)")
    print("=" * 70)

    print(f"\nLoading prompt model: {prompt_model_path.name}")
    prompt_graph = webnn.MLGraph.load(
        str(prompt_model_path),
        manifest_path=str(manifest_path),
        weights_path=str(weights_path),
    )
    print("Model loaded!")

    # Check what inputs the model expects and their shapes
    model_inputs = prompt_graph.get_input_names()
    kv_input_names = [name for name in model_inputs if name.startswith("past_key_values_")]

    # Check if KV inputs are empty (shape contains 0)
    # For clean models with past_sequence_length=0, KV inputs exist but have shape [1, 3, 0, 64]
    has_empty_kv = False
    if kv_input_names:
        # Check if we're using the clean model (heuristic based on filename)
        has_empty_kv = "clean" in str(prompt_model_path)

    has_kv_inputs = len(kv_input_names) > 0 and not has_empty_kv

    print(f"\nModel expects {len(model_inputs)} inputs")
    if has_empty_kv:
        print("  ✅ Model has empty KV cache inputs (past_sequence_length = 0)")
        past_seq_len = 0
        attention_mask_size = MAX_CACHE_SIZE
    elif has_kv_inputs:
        print("  ℹ️  Model has KV cache inputs (past_sequence_length > 0)")
        past_seq_len = 1  # Workaround model
        attention_mask_size = MAX_CACHE_SIZE + past_seq_len
    else:
        print("  ✅ Model has no KV cache inputs (past_sequence_length = 0)")
        past_seq_len = 0
        attention_mask_size = MAX_CACHE_SIZE

    # Prepare inputs - pad to sequence_length=128
    input_ids_padded = np.zeros((1, MAX_CACHE_SIZE), dtype=np.int64)
    input_ids_padded[0, :prompt_len] = prompt_tokens

    # Attention mask size depends on past_sequence_length
    attention_mask_prompt = np.zeros((1, attention_mask_size), dtype=np.int64)
    attention_mask_prompt[0, :prompt_len] = 1

    position_ids_padded = np.zeros((1, MAX_CACHE_SIZE), dtype=np.int64)
    position_ids_padded[0, :prompt_len] = list(range(prompt_len))

    # Build feed dict
    feed_prompt = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_prompt,
        "position_ids": position_ids_padded,
    }

    # Add KV cache inputs only if they're non-empty
    # Empty KV inputs will be removed by the converter, so don't provide them
    if has_kv_inputs:
        print(f"  Adding {num_layers * 2} KV cache inputs (shape: [1, {num_kv_heads}, {past_seq_len}, {head_dim}])")

        kv_cache_prompt = {}
        for layer in range(num_layers):
            kv_cache_prompt[f"past_key_values_{layer}_key"] = np.zeros(
                (1, num_kv_heads, past_seq_len, head_dim), dtype=np.float32
            )
            kv_cache_prompt[f"past_key_values_{layer}_value"] = np.zeros(
                (1, num_kv_heads, past_seq_len, head_dim), dtype=np.float32
            )
        feed_prompt.update(kv_cache_prompt)
    else:
        if has_empty_kv:
            print(f"  Empty KV cache inputs detected - converter will remove them automatically")
        else:
            print(f"  No KV cache inputs needed")

    if args.verbose:
        print(f"\nPrompt inputs:")
        print(f"  input_ids: {feed_prompt['input_ids'].shape}")
        print(f"  attention_mask: {feed_prompt['attention_mask'].shape}")
        print(f"  position_ids: {feed_prompt['position_ids'].shape}")

    print(f"\nProcessing {prompt_len} tokens as batch...")

    # Execute prompt model
    result_prompt = context.compute(prompt_graph, feed_prompt)

    # Extract logits and first generated token
    logits_prompt = result_prompt["logits"]  # [1, 128, vocab_size]

    # Get first generated token from last prompt position
    next_token_id = int(np.argmax(logits_prompt[0, prompt_len - 1, :]))
    first_token_text = tokenizer.decode([next_token_id])

    print(f"Prompt processed!")
    print(f"First generated token: '{first_token_text}' (id={next_token_id})")

    # Extract and pad KV cache for generation phase
    print(f"\nTransferring KV cache to generation model...")
    kv_cache_full = {}
    for layer in range(num_layers):
        present_key = result_prompt[f"present_{layer}_key"]
        present_value = result_prompt[f"present_{layer}_value"]

        if args.verbose and layer == 0:
            print(f"  present_key shape: {present_key.shape}")

        # Create full-sized cache arrays
        kv_cache_full[f"past_key_values_{layer}_key"] = np.zeros(
            (1, num_kv_heads, MAX_CACHE_SIZE, head_dim), dtype=np.float32
        )
        kv_cache_full[f"past_key_values_{layer}_value"] = np.zeros(
            (1, num_kv_heads, MAX_CACHE_SIZE, head_dim), dtype=np.float32
        )

        # Copy prompt cache
        # If model had past_sequence_length=1 workaround: shape is [1, 3, prompt_len+1, 64]
        # If model had past_sequence_length=0: shape is [1, 3, prompt_len, 64]
        if has_kv_inputs and past_seq_len > 0:
            # Workaround model: skip dummy position 0, take positions 1:prompt_len+1
            kv_cache_full[f"past_key_values_{layer}_key"][:, :, :prompt_len, :] = \
                present_key[:, :, 1:prompt_len+1, :]
            kv_cache_full[f"past_key_values_{layer}_value"][:, :, :prompt_len, :] = \
                present_value[:, :, 1:prompt_len+1, :]
        else:
            # Clean model: take all positions 0:prompt_len
            kv_cache_full[f"past_key_values_{layer}_key"][:, :, :prompt_len, :] = \
                present_key[:, :, :prompt_len, :]
            kv_cache_full[f"past_key_values_{layer}_value"][:, :, :prompt_len, :] = \
                present_value[:, :, :prompt_len, :]

    if args.verbose:
        print(f"  KV cache shape: [1, {num_kv_heads}, {MAX_CACHE_SIZE}, {head_dim}]")
        print(f"  Filled positions: 0-{prompt_len-1}")

    print("Phase 1 complete!")

    # =====================================================
    # PHASE 2: TOKEN-BY-TOKEN GENERATION
    # =====================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Token-by-Token Generation")
    print("=" * 70)

    print(f"\nLoading generation model: {gen_model_path.name}")
    gen_graph = webnn.MLGraph.load(
        str(gen_model_path),
        manifest_path=str(manifest_path),
        weights_path=str(weights_path),
    )
    print("Model loaded!")

    # Initialize attention mask for generation
    attention_mask_gen = np.zeros((1, MAX_CACHE_SIZE + 1), dtype=np.int64)
    attention_mask_gen[0, :prompt_len] = 1  # Enable prompt positions

    current_pos = prompt_len
    generated_tokens = [next_token_id]

    print(f"\nGenerating {args.max_new_tokens - 1} more tokens...")

    # Generate remaining tokens
    for step in range(args.max_new_tokens - 1):  # -1 because we have first token
        if current_pos >= MAX_CACHE_SIZE:
            print(f"\nReached cache limit ({MAX_CACHE_SIZE})")
            break

        # Enable attention for current position
        attention_mask_gen[0, current_pos] = 1

        # Build feed dict
        feed_gen = {
            "input_ids": np.array([[next_token_id]], dtype=np.int64),
            "attention_mask": attention_mask_gen,
            "position_ids": np.array([[current_pos]], dtype=np.int64),
            **kv_cache_full
        }

        # Execute generation model
        result_gen = context.compute(gen_graph, feed_gen)

        # Get next token
        logits_gen = result_gen["logits"]  # [1, 1, vocab_size]
        next_token_id = int(np.argmax(logits_gen[0, 0, :]))

        token_text = tokenizer.decode([next_token_id])
        generated_tokens.append(next_token_id)

        print(f"  Step {step + 1}: '{token_text}' (id={next_token_id})")

        # Update cache at current position
        for layer in range(num_layers):
            present_key = result_gen[f"present_{layer}_key"]  # [1, 3, current_pos+1, 64]
            present_value = result_gen[f"present_{layer}_value"]

            # Extract just the new position
            new_key = present_key[:, :, current_pos:current_pos+1, :]
            new_value = present_value[:, :, current_pos:current_pos+1, :]

            kv_cache_full[f"past_key_values_{layer}_key"][:, :, current_pos:current_pos+1, :] = new_key
            kv_cache_full[f"past_key_values_{layer}_value"][:, :, current_pos:current_pos+1, :] = new_value

        current_pos += 1

        # Stop if EOS
        if next_token_id == tokenizer.eos_token_id:
            print("\nReached EOS token")
            break

    print("Phase 2 complete!")

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
    print(f"Cache utilization: {current_pos}/{MAX_CACHE_SIZE} positions")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
