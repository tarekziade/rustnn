#!/usr/bin/env python3
"""
Regenerate /tmp/smollm_debug.onnx from the WebNN prompt model.
This is needed after fixing the output ordering bug.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import webnn
import numpy as np

MODEL_DIR = Path.home() / "Dev" / "SmolLM-135-webnn" / "prompt_model_clean"

def main():
    print("=" * 70)
    print("Regenerating /tmp/smollm_debug.onnx from WebNN prompt model")
    print("=" * 70)

    # Load WebNN model
    print("\nLoading WebNN prompt model...")
    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")

    # Load the graph - this will trigger conversion and save debug ONNX
    graph = webnn.MLGraph.load(
        str(MODEL_DIR / "smollm-135-prompt-clean.webnn"),
        str(MODEL_DIR / "smollm-135-prompt-clean.manifest.json"),
        str(MODEL_DIR / "smollm-135-prompt-clean.weights")
    )

    print("Graph loaded!")

    # Create dummy inputs with required shape [1, 128]
    # Just put 1 real token at the beginning, rest are padding
    input_ids = np.zeros((1, 128), dtype=np.int64)
    input_ids[0, 0] = 504  # First token

    attention_mask = np.zeros((1, 128), dtype=np.int64)
    attention_mask[0, 0] = 1  # Only first token is valid

    position_ids = np.zeros((1, 128), dtype=np.int64)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    print("Running compute to trigger ONNX conversion...")
    result = context.compute(graph, inputs)

    print("\n[OK] /tmp/smollm_debug.onnx has been regenerated with fixed output ordering!")
    print("Logits should now be at output 0.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
