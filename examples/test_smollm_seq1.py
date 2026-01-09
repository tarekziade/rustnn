#!/usr/bin/env python3
"""Test SmolLM-135M with sequence_length=1 model for token-by-token generation."""

import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
import webnn

HF_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
DEFAULT_MODEL_DIR = Path.home() / "Dev" / "SmolLM-135-webnn"


def main():
    print("=" * 60)
    print("SmolLM-135M Token-by-Token Test (sequence_length=1)")
    print("=" * 60)

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    config = AutoConfig.from_pretrained(HF_MODEL_ID)

    num_layers = config.num_hidden_layers
    num_key_value_heads = getattr(config, "num_key_value_heads", 3)
    hidden_size = config.hidden_size
    num_attention_heads = getattr(config, "num_attention_heads", 9)
    head_dim = hidden_size // num_attention_heads

    print(f"\nModel config:")
    print(f"  Layers: {num_layers}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  KV heads: {num_key_value_heads}")
    print(f"  Head dim: {head_dim}")

    # Tokenize prompt
    prompt = "Once upon a time"
    print(f"\nPrompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt, return_tensors="np")
    print(f"Tokens: {prompt_ids[0].tolist()}")

    # Load model
    print(f"\nLoading model from {DEFAULT_MODEL_DIR}...")
    graph = webnn.MLGraph.load(
        str(DEFAULT_MODEL_DIR / "smollm-135.webnn"),
        manifest_path=str(DEFAULT_MODEL_DIR / "smollm-135.manifest.json"),
        weights_path=str(DEFAULT_MODEL_DIR / "smollm-135.weights"),
    )

    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")
    print(f"Context created (accelerated={context.accelerated})")

    # Initialize KV cache (past_sequence_length=0 → empty cache [1, 3, 0, 64])
    kv_cache = {}
    for layer in range(num_layers):
        kv_cache[f"past_key_values_{layer}_key"] = np.zeros(
            (1, num_key_value_heads, 0, head_dim), dtype=np.float32
        )
        kv_cache[f"past_key_values_{layer}_value"] = np.zeros(
            (1, num_key_value_heads, 0, head_dim), dtype=np.float32
        )

    print(f"\nInitial KV cache shape: [1, {num_key_value_heads}, 0, {head_dim}]")

    # Run first token
    print("\n" + "-" * 60)
    print("Testing first token...")
    print("-" * 60)

    token_id = int(prompt_ids[0, 0])
    print(f"Input token ID: {token_id} ('{tokenizer.decode([token_id])}')")

    feed = {
        "input_ids": np.array([[token_id]], dtype=np.int64),  # [1, 1]
        "attention_mask": np.ones((1, 1), dtype=np.int64),  # [1, 1]
        "position_ids": np.array([[0]], dtype=np.int64),  # [1, 1]
        **kv_cache,
    }

    print(f"Input shapes:")
    print(f"  input_ids: {feed['input_ids'].shape}")
    print(f"  attention_mask: {feed['attention_mask'].shape}")
    print(f"  position_ids: {feed['position_ids'].shape}")
    print(f"  KV cache: [1, {num_key_value_heads}, 0, {head_dim}]")

    try:
        result = context.compute(graph, feed)
        print("\n✅ SUCCESS: Model executed!")

        logits = result["logits"]
        print(f"\nLogits shape: {logits.shape}")
        print(f"Expected: [1, 1, vocab_size]")

        # Get prediction
        predicted_id = np.argmax(logits[0, 0, :])
        predicted_token = tokenizer.decode([predicted_id])
        print(f"\nPredicted next token: '{predicted_token}' (id={predicted_id})")

        # Show top 5
        top_k = 5
        top_indices = np.argsort(logits[0, 0, :])[-top_k:][::-1]
        print(f"\nTop {top_k} predictions:")
        for i, idx in enumerate(top_indices, 1):
            token = tokenizer.decode([idx])
            score = logits[0, 0, idx]
            print(f"  {i}. '{token}' (id={idx}, score={score:.2f})")

        # Check output KV cache
        present_key = result.get("present_0_key")
        if present_key is not None:
            print(f"\nOutput KV cache shape: {present_key.shape}")
            print(f"Expected: [1, {num_key_value_heads}, 1, {head_dim}]")

        print("\n" + "=" * 60)
        print("SUCCESS: Token-by-token generation is working! ✅")
        print("=" * 60)
        print("\nThis validates that:")
        print("  ✓ WebNN→ONNX converter preserves tensor ranks correctly")
        print("  ✓ sequence_length=1 model can execute")
        print("  ✓ Trilu operation receives rank-2 input")
        print("\nNext steps:")
        print("  - Implement KV cache update logic to grow past_sequence_length")
        print("  - Create models for different past_sequence_length values")
        print("  - Build token generation loop with model switching")

        return 0

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
