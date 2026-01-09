#!/usr/bin/env python3
"""
Compare two ONNX models to find structural differences.
This helps identify what's causing the logit corruption in Bug #2.
"""
import sys
from pathlib import Path
import onnx
from collections import defaultdict, Counter

def analyze_model(model_path):
    """Analyze an ONNX model and return statistics."""
    model = onnx.load(model_path)
    graph = model.graph

    stats = {
        "path": model_path,
        "nodes": len(graph.node),
        "inputs": len(graph.input),
        "outputs": len(graph.output),
        "initializers": len(graph.initializer),
        "op_types": Counter(),
        "nodes_by_op": defaultdict(list),
        "final_nodes": [],  # Nodes that connect to outputs
    }

    # Count operation types
    for node in graph.node:
        stats["op_types"][node.op_type] += 1
        stats["nodes_by_op"][node.op_type].append(node.name)

    # Find nodes that directly feed into outputs
    output_names = {out.name for out in graph.output}
    for node in graph.node:
        if any(out in output_names for out in node.output):
            stats["final_nodes"].append({
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            })

    return stats, graph

def compare_models(original_path, converted_path):
    """Compare two ONNX models."""
    print("=" * 70)
    print("ONNX Graph Comparison: Finding Bug #2 Root Cause")
    print("=" * 70)

    print(f"\nLoading models...")
    orig_stats, orig_graph = analyze_model(original_path)
    conv_stats, conv_graph = analyze_model(converted_path)

    print(f"\nOriginal:  {original_path}")
    print(f"Converted: {converted_path}")

    # Compare basic stats
    print("\n" + "=" * 70)
    print("Basic Statistics")
    print("=" * 70)
    print(f"{'Metric':<20} {'Original':>15} {'Converted':>15} {'Difference':>15}")
    print("-" * 70)

    metrics = [
        ("Nodes", "nodes"),
        ("Inputs", "inputs"),
        ("Outputs", "outputs"),
        ("Initializers", "initializers"),
    ]

    for label, key in metrics:
        orig_val = orig_stats[key]
        conv_val = conv_stats[key]
        diff = conv_val - orig_val
        diff_str = f"{diff:+d}" if diff != 0 else "="
        print(f"{label:<20} {orig_val:>15} {conv_val:>15} {diff_str:>15}")

    # Compare operation types
    print("\n" + "=" * 70)
    print("Operation Type Comparison")
    print("=" * 70)
    print(f"{'Op Type':<20} {'Original':>15} {'Converted':>15} {'Difference':>15}")
    print("-" * 70)

    all_op_types = sorted(set(orig_stats["op_types"].keys()) | set(conv_stats["op_types"].keys()))
    for op_type in all_op_types:
        orig_count = orig_stats["op_types"][op_type]
        conv_count = conv_stats["op_types"][op_type]
        diff = conv_count - orig_count
        diff_str = f"{diff:+d}" if diff != 0 else "="

        # Highlight differences
        marker = ""
        if diff != 0:
            marker = " [DIFF]"

        print(f"{op_type:<20} {orig_count:>15} {conv_count:>15} {diff_str:>15}{marker}")

    # Find the logits output node
    print("\n" + "=" * 70)
    print("Nodes Producing 'logits' Output")
    print("=" * 70)

    print("\nOriginal model:")
    orig_logits_node = None
    for node in orig_graph.node:
        if "logits" in node.output:
            orig_logits_node = node
            print(f"  Node: {node.name}")
            print(f"  Op: {node.op_type}")
            print(f"  Inputs: {list(node.input)}")
            print(f"  Outputs: {list(node.output)}")
            break

    print("\nConverted model:")
    conv_logits_node = None
    for node in conv_graph.node:
        if "logits" in node.output:
            conv_logits_node = node
            print(f"  Node: {node.name}")
            print(f"  Op: {node.op_type}")
            print(f"  Inputs: {list(node.input)}")
            print(f"  Outputs: {list(node.output)}")
            break

    # Trace back from logits node
    if orig_logits_node and conv_logits_node:
        print("\n" + "=" * 70)
        print("Tracing Back from 'logits' Node (5 levels)")
        print("=" * 70)

        print("\nOriginal model:")
        trace_inputs(orig_graph, orig_logits_node.input, depth=5, indent=2)

        print("\nConverted model:")
        trace_inputs(conv_graph, conv_logits_node.input, depth=5, indent=2)

    # Check for suspicious patterns
    print("\n" + "=" * 70)
    print("Suspicious Patterns")
    print("=" * 70)

    # Check for doubled Add/MatMul/Gemm near the end
    suspicious = []

    # Check if MatMul/Gemm is duplicated
    if conv_stats["op_types"]["MatMul"] > orig_stats["op_types"]["MatMul"]:
        suspicious.append(f"MatMul operations increased: {orig_stats['op_types']['MatMul']} -> {conv_stats['op_types']['MatMul']}")

    if conv_stats["op_types"]["Add"] > orig_stats["op_types"]["Add"]:
        suspicious.append(f"Add operations increased: {orig_stats['op_types']['Add']} -> {conv_stats['op_types']['Add']}")

    if suspicious:
        for pattern in suspicious:
            print(f"  [WARNING] {pattern}")
    else:
        print("  No obvious duplications detected")

    print("\n" + "=" * 70)
    print("Recommendation")
    print("=" * 70)
    print("\nTo find the exact issue:")
    print("1. Compare the nodes traced back from 'logits' above")
    print("2. Look for operations that appear in converted but not original")
    print("3. Check if any MatMul/Add operations are duplicated near the end")
    print("4. Inspect the weight shapes of the final linear layer (lm_head)")

def trace_inputs(graph, input_names, depth=3, indent=0, seen=None):
    """Recursively trace input nodes."""
    if seen is None:
        seen = set()

    if depth == 0:
        return

    prefix = " " * indent
    for input_name in input_names:
        if input_name in seen:
            continue
        seen.add(input_name)

        # Find the node that produces this input
        producer_node = None
        for node in graph.node:
            if input_name in node.output:
                producer_node = node
                break

        if producer_node:
            print(f"{prefix}← {input_name} (from {producer_node.op_type}: {producer_node.name})")
            trace_inputs(graph, producer_node.input, depth - 1, indent + 2, seen)
        else:
            # It's an input or initializer
            print(f"{prefix}← {input_name} [input/weight]")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_onnx_graphs.py <original.onnx> <converted.onnx>")
        return 1

    original_path = sys.argv[1]
    converted_path = sys.argv[2]

    if not Path(original_path).exists():
        print(f"Error: {original_path} not found")
        return 1

    if not Path(converted_path).exists():
        print(f"Error: {converted_path} not found")
        return 1

    compare_models(original_path, converted_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
