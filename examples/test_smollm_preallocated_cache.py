#!/usr/bin/env python3
"""
Test SmolLM-135M with pre-allocated KV cache approach.

This demonstrates how to do token-by-token generation with WebNN's
static shape requirement by pre-allocating the full KV cache and
using attention masking.
"""

import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
import webnn

HF_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
DEFAULT_MODEL_DIR = Path.home() / "Dev" / "SmolLM-135-webnn"


def main():
    print("=" * 70)
    print("SmolLM-135M Pre-Allocated Cache Token-by-Token Generation")
    print("=" * 70)

    # Configuration
    MAX_CONTEXT_LENGTH = 128  # Pre-allocated cache size
    MAX_NEW_TOKENS = 10

    # Load tokenizer and config
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
    print(f"  Max context: {MAX_CONTEXT_LENGTH} tokens")

    # Calculate memory usage
    kv_memory_mb = (
        num_layers * 2 * 4 * 1 * num_key_value_heads * MAX_CONTEXT_LENGTH * head_dim
    ) / (1024 * 1024)
    print(f"  KV cache memory: {kv_memory_mb:.1f} MB")

    # Tokenize prompt
    prompt = "Once upon a time"
    print(f"\nPrompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt, return_tensors="np")
    prompt_tokens = prompt_ids[0].tolist()
    print(f"Tokens: {prompt_tokens} ({len(prompt_tokens)} tokens)")

    if len(prompt_tokens) > MAX_CONTEXT_LENGTH:
        print(f"\nERROR: Prompt exceeds max context length!")
        return 1

    # Load model (would need to be converted with past_sequence_length=MAX_CONTEXT_LENGTH)
    print(f"\nNOTE: This requires a model converted with:")
    print(f"  --override-dim past_sequence_length={MAX_CONTEXT_LENGTH}")
    print(f"  --override-dim \"past_sequence_length + 1\"={MAX_CONTEXT_LENGTH + 1}")
    print(f"  --override-dim sequence_length=1")
    print(f"\nCurrent model directory: {DEFAULT_MODEL_DIR}")

    # For demonstration, show the data structures
    print("\n" + "=" * 70)
    print("Data Structure Initialization")
    print("=" * 70)

    # Pre-allocate full KV cache
    print(f"\n1. Pre-allocating KV cache: [1, {num_key_value_heads}, {MAX_CONTEXT_LENGTH}, {head_dim}]")
    kv_cache = {}
    for layer in range(num_layers):
        kv_cache[f"past_key_values_{layer}_key"] = np.zeros(
            (1, num_key_value_heads, MAX_CONTEXT_LENGTH, head_dim),
            dtype=np.float32
        )
        kv_cache[f"past_key_values_{layer}_value"] = np.zeros(
            (1, num_key_value_heads, MAX_CONTEXT_LENGTH, head_dim),
            dtype=np.float32
        )

    print(f"   Total tensors: {len(kv_cache)}")
    print(f"   Memory per layer: {kv_cache['past_key_values_0_key'].nbytes / 1024:.1f} KB")

    # Initialize attention mask
    print(f"\n2. Initializing attention mask: [1, {MAX_CONTEXT_LENGTH + 1}]")
    attention_mask = np.zeros((1, MAX_CONTEXT_LENGTH + 1), dtype=np.int64)
    print(f"   All positions masked (0) initially")

    # Demonstrate the generation loop
    print("\n" + "=" * 70)
    print("Token-by-Token Generation Process")
    print("=" * 70)

    current_pos = 0
    generated_tokens = []

    for i, token_id in enumerate(prompt_tokens[:min(5, len(prompt_tokens))]):
        print(f"\n--- Step {i}: Processing token '{tokenizer.decode([token_id])}' (id={token_id}) ---")

        # Enable attention for current position
        attention_mask[0, current_pos] = 1
        print(f"Attention mask: {current_pos + 1} positions enabled")
        print(f"  Mask pattern: [{', '.join(str(attention_mask[0, j]) for j in range(min(10, current_pos + 5)))}{'...' if current_pos + 5 > 10 else ''}]")

        # Build feed dict
        feed = {
            "input_ids": np.array([[token_id]], dtype=np.int64),
            "attention_mask": attention_mask,
            "position_ids": np.array([[current_pos]], dtype=np.int64),
            **kv_cache
        }

        print(f"Feed dict prepared:")
        print(f"  input_ids: shape {feed['input_ids'].shape}")
        print(f"  attention_mask: shape {feed['attention_mask'].shape}, {np.sum(attention_mask)} enabled")
        print(f"  position_ids: [[{current_pos}]]")
        print(f"  KV cache: {num_layers * 2} tensors, shape [1, {num_key_value_heads}, {MAX_CONTEXT_LENGTH}, {head_dim}]")

        # In real execution:
        # result = context.compute(graph, feed)
        #
        # # Update cache at current position
        # for layer in range(num_layers):
        #     present_key = result[f"present_{layer}_key"]  # [1, 3, 1, 64]
        #     present_value = result[f"present_{layer}_value"]
        #
        #     kv_cache[f"past_key_values_{layer}_key"][:, :, current_pos:current_pos+1, :] = present_key
        #     kv_cache[f"past_key_values_{layer}_value"][:, :, current_pos:current_pos+1, :] = present_value

        print(f"Would update KV cache at position {current_pos}")
        print(f"  Cache slice: [:, :, {current_pos}:{current_pos+1}, :] = present_* output")

        generated_tokens.append(token_id)
        current_pos += 1

    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("\n✅ Advantages:")
    print("  • Works with WebNN's static shape requirement")
    print("  • True token-by-token generation (not batched)")
    print("  • Compatible with ONNX Runtime (no zero-sized dimensions)")
    print("  • Standard transformer attention masking")

    print("\n❌ Trade-offs:")
    print(f"  • High memory: {kv_memory_mb:.1f} MB for {MAX_CONTEXT_LENGTH}-token context")
    print("  • Wasted computation on unused cache positions")
    print(f"  • Fixed context limit: {MAX_CONTEXT_LENGTH} tokens maximum")

    print("\n💡 Memory Optimization:")
    print("  • Use smaller context for lower memory (e.g., 512, 256 tokens)")
    print("  • Trade-off: Less memory but shorter conversation history")

    print("\n🔧 Required Model Conversion:")
    print("  cd /path/to/webnn-wg")
    print(f"  ./target/release/webnn-graph convert-onnx \\")
    print(f"    --input smollm-135.onnx \\")
    print(f"    --output smollm-135-cached.webnn \\")
    print(f"    --override-dim past_sequence_length={MAX_CONTEXT_LENGTH} \\")
    print(f"    --override-dim \"past_sequence_length + 1\"={MAX_CONTEXT_LENGTH + 1} \\")
    print(f"    --override-dim sequence_length=1")

    print("\n" + "=" * 70)
    print("This approach enables LLM inference within WebNN's constraints!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
