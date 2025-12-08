#!/usr/bin/env python3
"""
WebNN Image Classification Demo
================================

This example demonstrates image classification using the WebNN Python API.
It shows how to:
- Load and preprocess images
- Build a neural network graph
- Run inference
- Display top predictions

Requirements:
    pip install pillow numpy

Usage:
    python examples/image_classification.py path/to/image.jpg
"""

import sys
import time
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install pillow numpy")
    sys.exit(1)

import webnn


# ImageNet class labels (1000 classes)
IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    # ... (truncated for brevity - full list would have 1000 classes)
    # In a real implementation, load from a JSON file
]


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for classification.

    Args:
        image_path: Path to the input image
        target_size: Target size for the image (height, width)

    Returns:
        Preprocessed image as numpy array with shape (1, 3, 224, 224)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Resize to target size
    img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension: (1, 3, 224, 224)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def build_simple_classifier(builder, num_classes=1000):
    """
    Build a simplified image classification network.

    This is a demonstration network that shows how to use WebNN operations.
    For real applications, use pretrained models.

    Args:
        builder: WebNN MLGraphBuilder instance
        num_classes: Number of output classes

    Returns:
        output_operand: The final output operand
    """
    # Input: (1, 3, 224, 224) - batch, channels, height, width
    input_tensor = builder.input("input", [1, 3, 224, 224], "float32")

    # Initial convolution: 3 -> 32 channels
    # In a real model, these weights would be loaded from a pretrained model
    conv1_weights = builder.constant(
        np.random.randn(32, 3, 3, 3).astype(np.float32) * 0.01,
        [32, 3, 3, 3],
        "float32"
    )
    conv1 = builder.conv2d(
        input_tensor,
        conv1_weights,
        padding=[1, 1, 1, 1],
        strides=[2, 2]
    )
    conv1 = builder.relu(conv1)

    # Depthwise convolution: 32 -> 32 (groups=32)
    dw_weights = builder.constant(
        np.random.randn(32, 1, 3, 3).astype(np.float32) * 0.01,
        [32, 1, 3, 3],
        "float32"
    )
    dw_conv = builder.conv2d(
        conv1,
        dw_weights,
        padding=[1, 1, 1, 1],
        groups=32
    )
    dw_conv = builder.relu(dw_conv)

    # Pointwise convolution: 32 -> 64
    pw_weights = builder.constant(
        np.random.randn(64, 32, 1, 1).astype(np.float32) * 0.01,
        [64, 32, 1, 1],
        "float32"
    )
    pw_conv = builder.conv2d(dw_conv, pw_weights)
    pw_conv = builder.relu(pw_conv)

    # Global average pooling
    pooled = builder.global_average_pool(pw_conv)

    # Reshape to (1, 64)
    flattened = builder.reshape(pooled, [1, 64])

    # Fully connected layer: 64 -> num_classes
    fc_weights = builder.constant(
        np.random.randn(num_classes, 64).astype(np.float32) * 0.01,
        [num_classes, 64],
        "float32"
    )
    logits = builder.matmul(flattened, builder.transpose(fc_weights, [1, 0]))

    # Softmax for class probabilities
    output = builder.softmax(logits)

    return output


def get_top_predictions(probabilities, top_k=5):
    """
    Get the top-k predictions from probability distribution.

    Args:
        probabilities: Numpy array of class probabilities
        top_k: Number of top predictions to return

    Returns:
        List of (class_index, probability) tuples
    """
    # Get indices of top-k probabilities
    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    # Create list of (index, probability) tuples
    top_predictions = [
        (int(idx), float(probabilities[idx]))
        for idx in top_indices
    ]

    return top_predictions


def main():
    """Main function to run image classification demo."""
    if len(sys.argv) < 2:
        print("Usage: python image_classification.py <image_path>")
        print("\nExample:")
        print("  python examples/image_classification.py examples/images/cat.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("WebNN Image Classification Demo")
    print("=" * 60)
    print(f"Image: {image_path}")
    print()

    # Load and preprocess image
    print("1. Loading and preprocessing image...")
    start_time = time.time()
    input_data = load_and_preprocess_image(image_path)
    preprocess_time = (time.time() - start_time) * 1000
    print(f"   ✓ Preprocessed to shape {input_data.shape} ({preprocess_time:.2f}ms)")
    print()

    # Create WebNN context
    print("2. Creating WebNN context...")
    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")
    print(f"   ✓ Context created (device: cpu)")
    print()

    # Build graph
    print("3. Building neural network graph...")
    start_time = time.time()
    builder = context.create_graph_builder()
    output = build_simple_classifier(builder, num_classes=1000)
    graph = builder.build({"output": output})
    build_time = (time.time() - start_time) * 1000
    print(f"   ✓ Graph built ({build_time:.2f}ms)")
    print()

    # Run inference
    print("4. Running inference...")
    start_time = time.time()
    results = context.compute(graph, {"input": input_data})
    inference_time = (time.time() - start_time) * 1000
    print(f"   ✓ Inference complete ({inference_time:.2f}ms)")
    print()

    # Get predictions
    output_probs = results["output"][0]  # Remove batch dimension
    top_predictions = get_top_predictions(output_probs, top_k=5)

    # Display results
    print("5. Top 5 Predictions:")
    print("-" * 60)
    for i, (class_idx, prob) in enumerate(top_predictions, 1):
        class_name = IMAGENET_CLASSES[class_idx] if class_idx < len(IMAGENET_CLASSES) else f"Class {class_idx}"
        print(f"   {i}. {class_name:30s} {prob*100:5.2f}%")
    print()

    print("=" * 60)
    print("Performance Summary:")
    print(f"  - Preprocessing: {preprocess_time:.2f}ms")
    print(f"  - Graph Build:   {build_time:.2f}ms")
    print(f"  - Inference:     {inference_time:.2f}ms")
    print("=" * 60)
    print()
    print("Note: This demo uses random weights for demonstration.")
    print("For real classification, load pretrained model weights.")


if __name__ == "__main__":
    main()
