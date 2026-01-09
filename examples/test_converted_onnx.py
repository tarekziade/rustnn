#!/usr/bin/env python3
"""
Test the WebNN-converted ONNX model with proper padding to 128 tokens.
This is Phase 1 of the investigation: Isolate Conversion vs Execution.
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
    parser = argparse.ArgumentParser(description="Test WebNN-converted ONNX model")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt",
    )
    parser.add_argument(
        "--converted-model",
        type=str,
        default="/tmp/smollm_debug.onnx",
        help="Path to converted ONNX model file",
    )
    parser.add_argument(
        "--original-model",
        type=str,
        default=str(Path.home() / "Dev" / "SmolLM-135-webnn" / "smollm-135.onnx"),
        help="Path to original ONNX model for comparison",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1: Test Converted ONNX with Proper Input Shapes")
    print("=" * 70)

    # Load tokenizer and config
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    config = AutoConfig.from_pretrained(HF_MODEL_ID)

    # Tokenize prompt
    print(f"\nPrompt: '{args.prompt}'")
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="np")
    prompt_tokens = prompt_ids[0].tolist()
    prompt_len = len(prompt_tokens)
    print(f"Tokens: {prompt_tokens} ({prompt_len} tokens)")

    # Load converted model
    print(f"\nLoading converted model: {args.converted_model}")
    converted_session = ort.InferenceSession(args.converted_model)

    # Get input shapes
    input_info = {inp.name: inp.shape for inp in converted_session.get_inputs()}
    print(f"\nConverted model inputs:")
    for name, shape in input_info.items():
        print(f"  {name}: {shape}")

    # Verify all inputs are [1, 128]
    expected_shape = [1, 128]
    for name, shape in input_info.items():
        if list(shape) != expected_shape:
            print(f"\nERROR: Expected all inputs to have shape {expected_shape}, but {name} has {shape}")
            return 1

    # Pad inputs to 128 tokens
    print(f"\nPadding inputs from {prompt_len} to {MAX_CACHE_SIZE} tokens...")

    # Pad input_ids (right padding with pad_token_id)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids_padded = np.full((1, MAX_CACHE_SIZE), pad_token_id, dtype=np.int64)
    input_ids_padded[0, :prompt_len] = prompt_tokens

    # Pad attention_mask (1 for valid tokens, 0 for padding)
    attention_mask_padded = np.zeros((1, MAX_CACHE_SIZE), dtype=np.int64)
    attention_mask_padded[0, :prompt_len] = 1

    # Pad position_ids (0 to prompt_len-1, then 0s for padding)
    position_ids_padded = np.zeros((1, MAX_CACHE_SIZE), dtype=np.int64)
    position_ids_padded[0, :prompt_len] = list(range(prompt_len))

    print(f"  input_ids: first {prompt_len} = {input_ids_padded[0, :prompt_len].tolist()}, rest = {pad_token_id}")
    print(f"  attention_mask: first {prompt_len} = 1, rest = 0")
    print(f"  position_ids: first {prompt_len} = [0..{prompt_len-1}], rest = 0")

    # Run converted model
    print(f"\nRunning converted model...")
    feed_dict = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "position_ids": position_ids_padded,
    }

    converted_outputs = converted_session.run(None, feed_dict)
    # After fix: logits should be at output 0
    converted_logits = converted_outputs[0]  # Logits at output 0 (fixed!)
    print(f"Converted logits shape: {converted_logits.shape}")

    # Get predicted token from converted model
    # The logits at position prompt_len-1 should be the prediction for the next token
    converted_next_token_id = int(np.argmax(converted_logits[0, prompt_len - 1, :]))
    converted_token_text = tokenizer.decode([converted_next_token_id])

    print(f"\nConverted model prediction:")
    print(f"  Next token: '{converted_token_text}' (id={converted_next_token_id})")
    print(f"  Top 5 tokens:")
    top5_indices = np.argsort(converted_logits[0, prompt_len - 1, :])[-5:][::-1]
    for idx in top5_indices:
        token_text = tokenizer.decode([idx])
        logit_value = converted_logits[0, prompt_len - 1, idx]
        print(f"    {idx}: '{token_text}' (logit={logit_value:.2f})")

    # Compare with original model
    print("\n" + "=" * 70)
    print("Comparing with Original Model")
    print("=" * 70)

    try:
        print(f"\nLoading original model: {args.original_model}")
        original_session = ort.InferenceSession(args.original_model)

        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", 3)
        hidden_size = config.hidden_size
        num_attention_heads = getattr(config, "num_attention_heads", 9)
        head_dim = hidden_size // num_attention_heads

        # Get input names
        input_names = [inp.name for inp in original_session.get_inputs()]
        kv_input_names = [name for name in input_names if "past_key_values" in name and name != "past_key_values"]
        has_kv_inputs = len(kv_input_names) > 0

        # Detect naming pattern
        uses_dots = False
        past_seq_len = 0
        if has_kv_inputs and len(kv_input_names) > 0:
            uses_dots = "." in kv_input_names[0]
            first_kv_input = [inp for inp in original_session.get_inputs() if inp.name == kv_input_names[0]][0]
            kv_shape = first_kv_input.shape
            if isinstance(kv_shape[2], int):
                past_seq_len = kv_shape[2]
            else:
                past_seq_len = 1

        print(f"Original model: {len(kv_input_names)} KV inputs, past_seq_len={past_seq_len}")

        # Prepare inputs for original model (no padding needed)
        input_ids = np.array([prompt_tokens], dtype=np.int64)
        attention_mask_size = prompt_len + past_seq_len
        attention_mask = np.zeros((1, attention_mask_size), dtype=np.int64)
        attention_mask[0, past_seq_len:past_seq_len + prompt_len] = 1
        position_ids = np.array([list(range(prompt_len))], dtype=np.int64)

        feed_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        # Add KV cache inputs
        if has_kv_inputs:
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

        print(f"Running original model...")
        original_outputs = original_session.run(None, feed_dict)
        original_logits = original_outputs[0]
        print(f"Original logits shape: {original_logits.shape}")

        # Get predicted token from original model
        original_next_token_id = int(np.argmax(original_logits[0, -1, :]))
        original_token_text = tokenizer.decode([original_next_token_id])

        print(f"\nOriginal model prediction:")
        print(f"  Next token: '{original_token_text}' (id={original_next_token_id})")
        print(f"  Top 5 tokens:")
        top5_indices = np.argsort(original_logits[0, -1, :])[-5:][::-1]
        for idx in top5_indices:
            token_text = tokenizer.decode([idx])
            logit_value = original_logits[0, -1, idx]
            print(f"    {idx}: '{token_text}' (logit={logit_value:.2f})")

        # Compare logits
        print("\n" + "=" * 70)
        print("Comparison Results")
        print("=" * 70)

        # Extract the last token logits from both models
        converted_last_logits = converted_logits[0, prompt_len - 1, :]
        original_last_logits = original_logits[0, -1, :]

        # Check if predictions match
        if converted_next_token_id == original_next_token_id:
            print(f"\n[OK] Predictions MATCH: '{original_token_text}' (id={original_next_token_id})")
        else:
            print(f"\n[FAIL] Predictions DIFFER:")
            print(f"  Original: '{original_token_text}' (id={original_next_token_id})")
            print(f"  Converted: '{converted_token_text}' (id={converted_next_token_id})")

        # Compare logit distributions
        max_diff = np.max(np.abs(converted_last_logits - original_last_logits))
        mean_diff = np.mean(np.abs(converted_last_logits - original_last_logits))
        print(f"\nLogit differences:")
        print(f"  Max absolute difference: {max_diff:.4f}")
        print(f"  Mean absolute difference: {mean_diff:.4f}")

        # Check if logits are close
        if np.allclose(converted_last_logits, original_last_logits, atol=1e-3):
            print(f"  [OK] Logits are close (within 1e-3 tolerance)")
        else:
            print(f"  [WARNING] Logits differ significantly")

        # Show logit values for top tokens from both models
        print(f"\nLogit values for top original tokens:")
        top5_original = np.argsort(original_last_logits)[-5:][::-1]
        for idx in top5_original:
            token_text = tokenizer.decode([idx])
            orig_logit = original_last_logits[idx]
            conv_logit = converted_last_logits[idx]
            diff = abs(orig_logit - conv_logit)
            print(f"  {idx} ('{token_text}'): orig={orig_logit:.2f}, conv={conv_logit:.2f}, diff={diff:.2f}")

    except Exception as e:
        print(f"\nError comparing with original model: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Phase 1 Complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
