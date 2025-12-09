#!/usr/bin/env python3
"""
Train the Text Generation Model
===============================

[WARNING]  NOTE: This is a TOY training script for DEMONSTRATION ONLY.
    It uses simple finite-difference gradients and won't produce
    quality results. The real purpose is to show how to save/load
    weights and demonstrate the training workflow concept.

    For real training, use proper frameworks (PyTorch, etc.) with
    backpropagation and real optimization algorithms.

Usage:
    python train_text_model.py --data shakespeare.txt --epochs 10 --save model.json
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Import the model from the generation script
sys.path.insert(0, str(Path(__file__).parent))
from text_generation_gpt import SimpleTransformerLM


def load_text_data(file_path):
    """Load text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def prepare_training_data(text, vocab_size=128, seq_len=32):
    """
    Convert text to training sequences (ASCII-only).

    Returns:
        X: input sequences (num_samples, seq_len)
        y: target tokens (num_samples,)
    """
    # Convert to ASCII bytes (0-127)
    data = np.array([b for b in text.encode('ascii', errors='ignore') if b < vocab_size], dtype=np.int32)

    if len(data) < seq_len + 1:
        raise ValueError(f"Text too short. Need at least {seq_len + 1} characters, got {len(data)}")

    # Create sequences
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])

    return np.array(X), np.array(y)


def compute_loss(model, X_batch, y_batch):
    """
    Compute cross-entropy loss for a batch.
    This is done in NumPy (not WebNN) for training.
    """
    batch_size, seq_len = X_batch.shape
    total_loss = 0

    for i in range(batch_size):
        # Get embeddings
        x_embedded = np.zeros((seq_len, model.d_model), dtype=np.float32)
        for j, token_id in enumerate(X_batch[i]):
            if token_id < model.vocab_size:
                x_embedded[j, :] = model.token_embed[token_id, :] + model.pos_embed[j, :]

        # Forward pass (simplified in NumPy)
        # Q, K, V
        Q = x_embedded @ model.W_q
        K = x_embedded @ model.W_k
        V = x_embedded @ model.W_v

        # Mean pooling approximation
        attn_output = np.mean(V, axis=0, keepdims=True)  # (1, d_model)

        # Layer norm (simplified)
        attn_output = (attn_output - np.mean(attn_output)) / (np.std(attn_output) + 1e-5)
        attn_output = attn_output * model.ln_gamma + model.ln_beta

        # Feed-forward
        h = attn_output @ model.W_ff1 + model.b_ff1
        h = np.maximum(h, 0)  # ReLU
        out = h @ model.W_ff2 + model.b_ff2

        # Layer norm again
        out = (out - np.mean(out)) / (np.std(out) + 1e-5)
        out = out * model.ln_gamma + model.ln_beta

        # Output logits
        logits = out @ model.W_out + model.b_out
        logits = logits[0]  # (vocab_size,)

        # Softmax
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        # Cross-entropy loss
        target = y_batch[i]
        if target < model.vocab_size:
            loss = -np.log(probs[target] + 1e-10)
            total_loss += loss

    return total_loss / batch_size


def train_step(model, X_batch, y_batch, lr=0.05):
    """
    Perform one training step with simple gradient descent.
    Improved to focus on most important parameters.
    """
    batch_size = X_batch.shape[0]
    epsilon = 0.0001

    # Get original loss
    loss = compute_loss(model, X_batch, y_batch)

    # Collect unique tokens in this batch
    unique_tokens = np.unique(np.concatenate([X_batch.flatten(), y_batch]))

    # Update embeddings for tokens that appear in this batch (more focused)
    for token_id in unique_tokens[:min(10, len(unique_tokens))]:
        if token_id < model.vocab_size:
            for dim in range(0, model.d_model, 4):  # Update every 4th dimension
                model.token_embed[token_id, dim] += epsilon
                loss_plus = compute_loss(model, X_batch, y_batch)
                model.token_embed[token_id, dim] -= epsilon

                grad = (loss_plus - loss) / epsilon
                model.token_embed[token_id, dim] -= lr * grad

    # Update output layer more aggressively (most important for predictions)
    # Focus on columns corresponding to target tokens
    for target in y_batch[:min(8, len(y_batch))]:
        if target < model.vocab_size:
            # Update this target's column in W_out
            for i in range(0, model.d_model, 8):  # Every 8th row
                model.W_out[i, target] += epsilon
                loss_plus = compute_loss(model, X_batch, y_batch)
                model.W_out[i, target] -= epsilon

                grad = (loss_plus - loss) / epsilon
                model.W_out[i, target] -= lr * grad

            # Update bias for this target
            model.b_out[target] += epsilon
            loss_plus = compute_loss(model, X_batch, y_batch)
            model.b_out[target] -= epsilon

            grad = (loss_plus - loss) / epsilon
            model.b_out[target] -= lr * grad

    # Update FFN biases (often important)
    for i in range(0, len(model.b_ff1), 8):
        model.b_ff1[i] += epsilon
        loss_plus = compute_loss(model, X_batch, y_batch)
        model.b_ff1[i] -= epsilon

        grad = (loss_plus - loss) / epsilon
        model.b_ff1[i] -= lr * grad

    for i in range(0, len(model.b_ff2), 8):
        model.b_ff2[i] += epsilon
        loss_plus = compute_loss(model, X_batch, y_batch)
        model.b_ff2[i] -= epsilon

        grad = (loss_plus - loss) / epsilon
        model.b_ff2[i] -= lr * grad

    return loss


def train(model, X, y, epochs=10, batch_size=32, lr=0.01):
    """Train the model."""
    num_samples = len(X)
    num_batches = num_samples // batch_size

    print(f"\nTraining on {num_samples} samples ({num_batches} batches per epoch)")
    print("=" * 70)

    for epoch in range(epochs):
        total_loss = 0
        indices = np.random.permutation(num_samples)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch_indices = indices[start_idx:end_idx]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss = train_step(model, X_batch, y_batch, lr=lr)
            total_loss += loss

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {avg_loss:.4f}", end='\r')

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} completed. Average loss: {avg_loss:.4f}                    ")

    print("=" * 70)
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train text generation model")
    parser.add_argument("--data", required=True, help="Text file to train on")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--save", required=True, help="Output file for weights (JSON)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    print("=" * 70)
    print("Training SimpleTransformerLM")
    print("=" * 70)
    print(f"Data file: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()

    # Load data
    print("Loading text data...")
    text = load_text_data(args.data)
    print(f"[OK] Loaded {len(text)} characters")
    print()

    # Prepare training data
    print("Preparing training sequences (ASCII-only)...")
    X, y = prepare_training_data(text, vocab_size=128, seq_len=32)
    print(f"[OK] Created {len(X)} training samples")
    print()

    # Initialize model
    print("Initializing model (ASCII vocab)...")
    model = SimpleTransformerLM(vocab_size=128, d_model=64, max_seq_len=32)
    print("[OK] Model initialized")

    # Train
    train(model, X, y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # Save weights
    print()
    model.save_weights(args.save)
    print()

    print("=" * 70)
    print("Training complete! You can now use the trained model:")
    print(f"  python examples/text_generation_gpt.py --weights {args.save} --prompt \"Your text\" --tokens 50")
    print("=" * 70)


if __name__ == "__main__":
    main()
