#!/usr/bin/env python3
"""
GenAI KV Cache Demo with Device Tensors

This example demonstrates zero-copy execution using device-resident tensors
for iterative GenAI workloads. It simulates a simplified transformer decoder
with KV cache, showing how device tensors eliminate host round-trips.

The demo compares two approaches:
1. Host tensors (MLTensor) - requires device-to-host copies each step
2. Device tensors (MLDeviceTensor) - keeps data on device across steps
"""

import sys
import time
import numpy as np

try:
    import webnn
except ImportError:
    print("Error: webnn package not found")
    print("Build and install with: make python-dev")
    sys.exit(1)


def build_decoder_graph(ctx, batch_size=1, seq_len=128, hidden_dim=256, num_heads=4):
    """
    Build a simplified KV cache update graph.

    The graph takes:
    - past_key: [batch, num_heads, seq_len, head_dim] - cached keys
    - past_value: [batch, num_heads, seq_len, head_dim] - cached values

    Returns:
    - present_key: [batch, num_heads, seq_len, head_dim] - updated keys
    - present_value: [batch, num_heads, seq_len, head_dim] - updated values
    """
    builder = ctx.create_graph_builder()
    head_dim = hidden_dim // num_heads

    # Inputs
    past_key = builder.input("past_key", [batch_size, num_heads, seq_len, head_dim], "float32")
    past_value = builder.input("past_value", [batch_size, num_heads, seq_len, head_dim], "float32")

    # For demo purposes: simulate KV cache update with small increment
    # present_key = past_key + 0.01
    # present_value = past_value + 0.01
    scale = builder.constant(np.full([batch_size, num_heads, seq_len, head_dim], 0.01, dtype=np.float32))
    present_key = builder.add(past_key, scale)
    present_value = builder.add(past_value, scale)

    # Build graph
    graph = builder.build({
        "present_key": present_key,
        "present_value": present_value,
    })

    return graph


def run_with_host_tensors(ctx, graph, num_steps=50):
    """Run decode loop with host tensors (baseline with copies)"""
    batch_size = 1
    seq_len = 128
    hidden_dim = 256
    num_heads = 4
    head_dim = hidden_dim // num_heads

    print(f"Running {num_steps} steps with HOST tensors...")

    # Initialize KV cache (host tensors for this baseline)
    kv_shape = [batch_size, num_heads, seq_len, head_dim]
    past_key = ctx.create_host_tensor(kv_shape, "float32")
    past_value = ctx.create_host_tensor(kv_shape, "float32")
    present_key = ctx.create_host_tensor(kv_shape, "float32")
    present_value = ctx.create_host_tensor(kv_shape, "float32")

    # Initialize with zeros
    ctx.write_tensor(past_key, np.zeros(kv_shape, dtype=np.float32))
    ctx.write_tensor(past_value, np.zeros(kv_shape, dtype=np.float32))

    start = time.time()

    for step in range(num_steps):
        # Execute graph
        ctx.dispatch(
            graph,
            {
                "past_key": past_key,
                "past_value": past_value,
            },
            {
                "present_key": present_key,
                "present_value": present_value,
            }
        )

        # Swap ping-pong buffers
        past_key, present_key = present_key, past_key
        past_value, present_value = present_value, past_value

    elapsed = time.time() - start

    # Read final output to verify
    final_key = ctx.read_tensor(past_key)

    print(f"  Completed in {elapsed:.3f}s ({num_steps/elapsed:.1f} steps/sec)")
    print(f"  Final key shape: {final_key.shape}")
    print(f"  Final key mean: {final_key.mean():.6f}, std: {final_key.std():.6f}")

    return elapsed


def run_with_device_tensors(ctx, graph, num_steps=50):
    """Run decode loop with device tensors (zero-copy)"""
    batch_size = 1
    seq_len = 128
    hidden_dim = 256
    num_heads = 4
    head_dim = hidden_dim // num_heads

    print(f"\nRunning {num_steps} steps with DEVICE tensors...")

    try:
        # Initialize KV cache (device tensors)
        kv_shape = [batch_size, num_heads, seq_len, head_dim]
        past_key_device = ctx.create_device_tensor(graph, kv_shape, "float32")
        past_value_device = ctx.create_device_tensor(graph, kv_shape, "float32")
        present_key_device = ctx.create_device_tensor(graph, kv_shape, "float32")
        present_value_device = ctx.create_device_tensor(graph, kv_shape, "float32")

        # Initialize with zeros (initial write to device)
        past_key_device.write(np.zeros(kv_shape, dtype=np.float32))
        past_value_device.write(np.zeros(kv_shape, dtype=np.float32))

        print(f"  Device tensors created:")
        print(f"    KV cache: {past_key_device}")

        start = time.time()

        for step in range(num_steps):
            # Execute graph with device tensors
            # This reads from device tensors and writes to device tensors
            # No host round-trips for KV cache!
            ctx.dispatch(
                graph,
                {
                    "past_key": past_key_device,
                    "past_value": past_value_device,
                },
                {
                    "present_key": present_key_device,
                    "present_value": present_value_device,
                }
            )

            # Swap ping-pong buffers (just swap references, no copies!)
            past_key_device, present_key_device = present_key_device, past_key_device
            past_value_device, present_value_device = present_value_device, past_value_device

        elapsed = time.time() - start

        # Read final output only once at the end
        final_key = past_key_device.read()

        print(f"  Completed in {elapsed:.3f}s ({num_steps/elapsed:.1f} steps/sec)")
        print(f"  Final key shape: {final_key.shape}")
        print(f"  Final key mean: {final_key.mean():.6f}, std: {final_key.std():.6f}")

        # Cleanup
        past_key_device.destroy()
        past_value_device.destroy()
        present_key_device.destroy()
        present_value_device.destroy()

        return elapsed

    except Exception as e:
        print(f"  Device tensor execution failed: {e}")
        print(f"  Note: This may be expected if device tensors are not fully supported yet")
        return None


def main():
    print("=" * 70)
    print("GenAI KV Cache Demo - Device Tensors vs Host Tensors")
    print("=" * 70)
    print()

    # Create context
    ml = webnn.ML()
    ctx = ml.create_context(device_type="cpu", accelerated=False)

    print(f"Context: {ctx}")
    print(f"Backend: {ctx.backend_info()}")
    print()

    # Build decoder graph
    print("Building decoder graph...")
    graph = build_decoder_graph(ctx)
    print(f"  Graph built successfully")
    print()

    # Configuration
    num_steps = 50

    print(f"Configuration:")
    print(f"  Batch size: 1")
    print(f"  Sequence length: 128")
    print(f"  Hidden dim: 256")
    print(f"  Num heads: 4")
    print(f"  Decode steps: {num_steps}")
    print()
    print("-" * 70)
    print()

    # Run with host tensors
    host_time = run_with_host_tensors(ctx, graph, num_steps)

    # Run with device tensors
    device_time = run_with_device_tensors(ctx, graph, num_steps)

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Host tensors:   {host_time:.3f}s ({num_steps/host_time:.1f} steps/sec)")
    if device_time:
        print(f"Device tensors: {device_time:.3f}s ({num_steps/device_time:.1f} steps/sec)")
        speedup = host_time / device_time
        print()
        print(f"Speedup: {speedup:.2f}x")
        if speedup > 1.0:
            print(f"Device tensors are {speedup:.2f}x FASTER")
        else:
            print(f"Device tensors are {1/speedup:.2f}x SLOWER")
        print()
        print("Note: Current implementation does host round-trips for both modes.")
        print("True zero-copy execution will show larger speedups on GPU/NPU.")
    else:
        print(f"Device tensors: Not available")
    print()
    print("This demo shows the API for device tensors.")
    print("Full zero-copy execution coming soon!")


if __name__ == "__main__":
    main()
