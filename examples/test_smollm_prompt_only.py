#!/usr/bin/env python3
"""
Test SmolLM-135M WebNN model in prompt-processing mode only.

This script validates that the sequence_length=128 model can successfully
process a padded prompt and produce logits. It does NOT attempt incremental
generation, which would require a different model with sequence_length=1.
"""

import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import webnn

HF_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
DEFAULT_MODEL_DIR = Path.home() / "Dev" / "SmolLM-135-webnn"


def main():
    print("=" * 60)
    print("SmolLM-135M WebNN Prompt Processing Test")
    print("=" * 60)

    # Load tokenizer
    print(f"\n[1/5] Loading tokenizer from {HF_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    config = AutoConfig.from_pretrained(HF_MODEL_ID)

    # Get model info
    num_layers = config.num_hidden_layers
    num_attention_heads = getattr(config, "num_attention_heads", 9)
    num_key_value_heads = getattr(config, "num_key_value_heads", 3)
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_attention_heads

    print(f"  Model config: {num_layers} layers, {num_attention_heads} attn heads, {num_key_value_heads} KV heads, head_dim={head_dim}")

    # Tokenize prompt
    prompt = "Once upon a time"
    print(f"\n[2/5] Tokenizing prompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt, return_tensors="np")
    prompt_len = prompt_ids.shape[1]
    print(f"  Prompt tokens: {prompt_ids}")
    print(f"  Prompt length: {prompt_len}")

    # Pad to sequence_length=128
    seq_len = 128
    if prompt_len < seq_len:
        pad_len = seq_len - prompt_len
        padding = np.full((1, pad_len), tokenizer.pad_token_id or 0, dtype=np.int64)
        padded_ids = np.concatenate([prompt_ids, padding], axis=1)
        print(f"  Padded to {seq_len} tokens (added {pad_len} pad tokens)")
    elif prompt_len > seq_len:
        padded_ids = prompt_ids[:, :seq_len]
        print(f"  Truncated to {seq_len} tokens")
    else:
        padded_ids = prompt_ids

    print(f"  Input shape: {padded_ids.shape}")

    # Load WebNN model
    print(f"\n[3/5] Loading WebNN model from {DEFAULT_MODEL_DIR}...")
    model_path = DEFAULT_MODEL_DIR / "smollm-135.webnn"
    weights_path = DEFAULT_MODEL_DIR / "smollm-135.weights"
    manifest_path = DEFAULT_MODEL_DIR / "smollm-135.manifest.json"

    graph = webnn.MLGraph.load(
        str(model_path),
        manifest_path=str(manifest_path),
        weights_path=str(weights_path),
    )
    print(f"  Model loaded: {model_path.name}")

    # Create context
    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")
    print(f"  Context created (accelerated={context.accelerated})")

    # Build feed dict
    print("\n[4/5] Preparing inputs...")
    feed = {
        "input_ids": padded_ids,
        "attention_mask": np.ones((1, 1), dtype=np.int64),  # [1,1] for past_sequence_length=0
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    }

    # Initialize empty KV caches (past_sequence_length=0)
    for layer_idx in range(num_layers):
        # Shape: [batch_size, num_kv_heads, past_sequence_length, head_dim]
        # With past_sequence_length=0, this is [1, 3, 0, 64]
        feed[f"past_key_values_{layer_idx}_key"] = np.zeros((1, num_key_value_heads, 0, head_dim), dtype=np.float32)
        feed[f"past_key_values_{layer_idx}_value"] = np.zeros((1, num_key_value_heads, 0, head_dim), dtype=np.float32)

    print(f"  Input IDs shape: {feed['input_ids'].shape}")
    print(f"  Attention mask shape: {feed['attention_mask'].shape}")
    print(f"  Position IDs shape: {feed['position_ids'].shape}")
    print(f"  KV cache shape (per layer): {feed['past_key_values_0_key'].shape}")
    print(f"  Total inputs: {len(feed)}")

    # Run inference
    print("\n[5/5] Running inference...")
    try:
        result = context.compute(graph, feed)
        print("  ✅ Inference completed successfully!")

        # Analyze outputs
        logits = result.get("logits")
        if logits is not None:
            print(f"\n  Logits shape: {logits.shape}")
            print(f"  Expected: [1, {seq_len}, vocab_size]")

            # Get predictions for the last actual token (before padding)
            last_token_logits = logits[0, prompt_len - 1, :]
            predicted_token_id = np.argmax(last_token_logits)
            predicted_token = tokenizer.decode([predicted_token_id])

            print(f"\n  Last token position: {prompt_len - 1}")
            print(f"  Predicted next token ID: {predicted_token_id}")
            print(f"  Predicted next token: '{predicted_token}'")

            # Show top 5 predictions
            top_k = 5
            top_indices = np.argsort(last_token_logits)[-top_k:][::-1]
            print(f"\n  Top {top_k} predictions:")
            for i, idx in enumerate(top_indices, 1):
                token = tokenizer.decode([idx])
                prob = np.exp(last_token_logits[idx]) / np.sum(np.exp(last_token_logits))
                print(f"    {i}. '{token}' (id={idx}, prob={prob:.4f})")

        # Check KV cache outputs
        present_key = result.get("present_0_key")
        if present_key is not None:
            print(f"\n  Present KV cache shape: {present_key.shape}")
            print(f"  Expected: [1, {num_key_value_heads}, {seq_len}, {head_dim}]")

        print("\n" + "=" * 60)
        print("SUCCESS: Prompt processing validated! ✅")
        print("=" * 60)
        print("\nNOTE: This test validates prompt processing only.")
        print("Incremental token generation would require a different")
        print("model with sequence_length=1 and growing past_sequence_length.")

        return 0

    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
