#!/usr/bin/env python3
"""
Next-Token Generation with Attention and WebNN
==============================================

This implements a simplified transformer-style model for next-token prediction
using WebNN operations. It demonstrates:
- Scaled dot-product attention
- Layer normalization
- Feed-forward networks
- Autoregressive token generation

The model generates one token at a time (like the JavaScript LLM demo),
using pre-trained or initialized weights.
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import webnn


class SimpleTransformerLM:
    """
    Simplified transformer language model for next-token generation.
    Uses WebNN operations to build the computational graph.
    """

    def __init__(self, vocab_size=256, d_model=64, max_seq_len=32):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Initialize weights (would be pre-trained in production)
        np.random.seed(42)

        # Token embeddings
        self.token_embed = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

        # Positional embeddings
        self.pos_embed = self._create_positional_embeddings(max_seq_len, d_model)

        # Attention weights (simplified single-head)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

        # Feed-forward weights
        self.W_ff1 = np.random.randn(d_model, d_model * 2).astype(np.float32) * 0.02
        self.b_ff1 = np.zeros((d_model * 2,), dtype=np.float32)
        self.W_ff2 = np.random.randn(d_model * 2, d_model).astype(np.float32) * 0.02
        self.b_ff2 = np.zeros((d_model,), dtype=np.float32)

        # Output projection
        self.W_out = np.random.randn(d_model, vocab_size).astype(np.float32) * 0.02
        self.b_out = np.zeros((vocab_size,), dtype=np.float32)

        # Layer norm parameters
        self.ln_gamma = np.ones((d_model,), dtype=np.float32)
        self.ln_beta = np.zeros((d_model,), dtype=np.float32)

    def _create_positional_embeddings(self, max_len, d_model):
        """Create sinusoidal positional embeddings."""
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def load_weights(self, weights_file):
        """Load pre-trained weights."""
        print(f"Loading weights from {weights_file}...")
        with open(weights_file, 'r') as f:
            weights = json.load(f)

        # Load all weights from JSON
        for key in ['token_embed', 'W_q', 'W_k', 'W_v', 'W_ff1', 'b_ff1',
                    'W_ff2', 'b_ff2', 'W_out', 'b_out', 'ln_gamma', 'ln_beta']:
            if key in weights:
                setattr(self, key, np.array(weights[key], dtype=np.float32))

        print("✓ Weights loaded")

    def save_weights(self, weights_file):
        """Save model weights."""
        print(f"Saving weights to {weights_file}...")
        weights = {
            'token_embed': self.token_embed.tolist(),
            'W_q': self.W_q.tolist(),
            'W_k': self.W_k.tolist(),
            'W_v': self.W_v.tolist(),
            'W_ff1': self.W_ff1.tolist(),
            'b_ff1': self.b_ff1.tolist(),
            'W_ff2': self.W_ff2.tolist(),
            'b_ff2': self.b_ff2.tolist(),
            'W_out': self.W_out.tolist(),
            'b_out': self.b_out.tolist(),
            'ln_gamma': self.ln_gamma.tolist(),
            'ln_beta': self.ln_beta.tolist()
        }

        with open(weights_file, 'w') as f:
            json.dump(weights, f)

        print("✓ Weights saved")

    def build_graph(self, builder, x_input):
        """
        Build the transformer forward pass using WebNN operations.
        x_input: (seq_len, d_model) - embedded input sequence
        Returns: logits (vocab_size,) - next token probabilities
        """
        # Create weight constants
        W_q = builder.constant(self.W_q, list(self.W_q.shape), "float32")
        W_k = builder.constant(self.W_k, list(self.W_k.shape), "float32")
        W_v = builder.constant(self.W_v, list(self.W_v.shape), "float32")

        W_ff1 = builder.constant(self.W_ff1, list(self.W_ff1.shape), "float32")
        b_ff1 = builder.constant(self.b_ff1.reshape(1, -1), [1, self.d_model * 2], "float32")
        W_ff2 = builder.constant(self.W_ff2, list(self.W_ff2.shape), "float32")
        b_ff2 = builder.constant(self.b_ff2.reshape(1, -1), [1, self.d_model], "float32")

        W_out = builder.constant(self.W_out, list(self.W_out.shape), "float32")
        b_out = builder.constant(self.b_out.reshape(1, -1), [1, self.vocab_size], "float32")

        ln_gamma = builder.constant(self.ln_gamma, list(self.ln_gamma.shape), "float32")
        ln_beta = builder.constant(self.ln_beta, list(self.ln_beta.shape), "float32")

        # Simplified attention: use last token's representation
        # Q, K, V = x @ W_q, x @ W_k, x @ W_v
        Q = builder.matmul(x_input, W_q)  # (seq_len, d_model)
        K = builder.matmul(x_input, W_k)  # (seq_len, d_model)
        V = builder.matmul(x_input, W_v)  # (seq_len, d_model)

        # Compute attention scores for last position
        # scores = Q[-1] @ K^T  (simplified to use reduce_mean for demo)
        # For simplicity, we'll use mean pooling over the sequence
        # In production, you'd implement proper causal attention

        # Mean pooling approximation of attention
        attn_output = builder.reduce_mean(V, axes=[0], keep_dimensions=False)  # (d_model,)
        attn_output = builder.reshape(attn_output, [1, self.d_model])

        # Layer norm (using built-in operation)
        x = builder.layer_normalization(attn_output, scale=ln_gamma, bias=ln_beta)

        # Feed-forward network
        # h = relu(x @ W_ff1 + b_ff1)
        h = builder.gemm(x, W_ff1)
        h = builder.add(h, b_ff1)
        h = builder.relu(h)

        # out = h @ W_ff2 + b_ff2
        out = builder.gemm(h, W_ff2)
        out = builder.add(out, b_ff2)

        # Another layer norm
        out = builder.layer_normalization(out, scale=ln_gamma, bias=ln_beta)

        # Output projection to vocabulary
        logits = builder.gemm(out, W_out)
        logits = builder.add(logits, b_out)

        # Apply softmax to get probabilities
        probs = builder.softmax(logits)

        return probs

    def generate(self, context, prompt_tokens, max_new_tokens=50, temperature=0.8):
        """
        Generate tokens autoregressively (like the JS LLM demo).

        Args:
            context: WebNN MLContext
            prompt_tokens: Initial token IDs
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature

        Returns:
            List of all token IDs (prompt + generated)
        """
        generated = list(prompt_tokens)

        print(f"\nGenerating {max_new_tokens} tokens autoregressively...")
        print("=" * 70)

        for step in range(max_new_tokens):
            # Get recent context (last max_seq_len tokens)
            context_tokens = generated[-self.max_seq_len:]
            seq_len = len(context_tokens)

            # Embed tokens (token + positional embeddings)
            x_embedded = np.zeros((seq_len, self.d_model), dtype=np.float32)
            for i, token_id in enumerate(context_tokens):
                if token_id < self.vocab_size:
                    x_embedded[i, :] = self.token_embed[token_id, :] + self.pos_embed[i, :]

            # Build WebNN graph
            builder = context.create_graph_builder()
            x_input = builder.input("x", [seq_len, self.d_model], "float32")
            probs_op = self.build_graph(builder, x_input)
            graph = builder.build({"probs": probs_op})

            # Run inference
            results = context.compute(graph, {"x": x_embedded})
            probs = results["probs"][0]  # (vocab_size,)

            # Apply temperature and sample
            if temperature > 0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)
                next_token = np.random.choice(self.vocab_size, p=probs)
            else:
                next_token = np.argmax(probs)

            generated.append(int(next_token))

            # Progress indicator
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Token {step + 1}/{max_new_tokens}: {next_token} (prob: {probs[next_token]:.4f})")

        print("=" * 70)
        return generated


def main():
    parser = argparse.ArgumentParser(
        description="Next-token generation with attention and WebNN"
    )
    parser.add_argument("--prompt", default="Hello world", help="Input text prompt")
    parser.add_argument("--tokens", type=int, default=30, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--weights", help="Pre-trained weights file (JSON)")
    parser.add_argument("--save-weights", help="Save weights to file")
    parser.add_argument("--backend", choices=["cpu", "gpu", "coreml"], default="cpu")
    args = parser.parse_args()

    # Backend configuration
    if args.backend == "cpu":
        accelerated, power, backend_name = False, "default", "ONNX CPU"
    elif args.backend == "gpu":
        accelerated, power, backend_name = True, "high-performance", "ONNX GPU"
    else:
        accelerated, power, backend_name = True, "low-power", "CoreML"

    print("=" * 70)
    print("Next-Token Generation with Attention (WebNN)")
    print("=" * 70)
    print(f"Backend: {backend_name}")
    print(f"Model: vocab=128 (ASCII), d_model=64, max_seq=32")
    print()

    # Create WebNN context
    print("Creating WebNN context...")
    ml = webnn.ML()
    context = ml.create_context(power_preference=power, accelerated=accelerated)
    print(f"✓ Context created (accelerated={context.accelerated})")
    print()

    # Initialize model
    print("Initializing transformer model...")
    model = SimpleTransformerLM(vocab_size=128, d_model=64, max_seq_len=32)
    print("✓ Model initialized (ASCII-only vocab for readable output)")
    print()

    # Load weights if provided
    if args.weights and Path(args.weights).exists():
        model.load_weights(args.weights)
        print()

    # Save weights if requested
    if args.save_weights:
        model.save_weights(args.save_weights)
        print()

    # Convert prompt to ASCII tokens (0-127 only)
    prompt_bytes = args.prompt.encode('ascii', errors='ignore')
    prompt_tokens = [b for b in prompt_bytes if b < 128]

    print(f"Prompt: '{args.prompt}'")
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens[:10]}{'...' if len(prompt_tokens) > 10 else ''}")
    print()

    # Generate tokens
    all_tokens = model.generate(
        context,
        prompt_tokens,
        max_new_tokens=args.tokens,
        temperature=args.temperature
    )

    # Decode to text
    try:
        generated_text = bytes(all_tokens).decode('utf-8', errors='replace')
    except:
        generated_text = str(all_tokens)

    print()
    print("Generated Text:")
    print("-" * 70)
    print(generated_text)
    print("-" * 70)
    print()

    print("=" * 70)
    print("WebNN Operations Demonstrated:")
    print("  ✓ matmul - Matrix multiplication for projections")
    print("  ✓ layer_normalization - Normalizing activations")
    print("  ✓ relu - Activation function")
    print("  ✓ softmax - Output probability distribution")
    print("  ✓ reduce_mean - Simplified attention pooling")
    print("  ✓ gemm - General matrix multiply with transpose")
    print()
    print("Next-Token Generation:")
    print("  ✓ Autoregressive generation (one token at a time)")
    print("  ✓ Attention mechanism (simplified)")
    print("  ✓ Temperature-based sampling")
    print("  ✓ Context window management")
    print()
    print("Note: Without pre-trained weights, output is random.")
    print("Train a model on real data and use --weights to load it!")
    print("=" * 70)


if __name__ == "__main__":
    main()
