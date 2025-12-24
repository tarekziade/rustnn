"""
Example: Save and Load WebNN Graphs

This example demonstrates:
1. Building a computational graph using the WebNN API
2. Saving the graph to a .webnn JSON file
3. Loading the graph back from the file
4. Executing both the original and loaded graphs to verify they work identically
"""

import webnn
import numpy as np
import os

def main():
    print("WebNN Graph Save/Load Example")
    print("=" * 50)

    # Create context and builder
    print("\n1. Creating WebNN context...")
    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")
    builder = context.create_graph_builder()

    # Build a simple graph: output = relu(matmul(x, W) + b)
    # This is a simple linear layer with ReLU activation
    print("\n2. Building graph (linear layer with ReLU)...")

    # Define inputs
    x = builder.input("x", [1, 4], "float32")  # Input: batch_size=1, features=4

    # Define constants (weights and bias)
    W = builder.constant(np.array([
        [0.5, -0.3],
        [0.2, 0.8],
        [-0.4, 0.6],
        [0.9, -0.1]
    ], dtype=np.float32), [4, 2], "float32")

    b = builder.constant(np.array([0.1, -0.2], dtype=np.float32), [2], "float32")

    # Build computation: matmul + add + relu
    z1 = builder.matmul(x, W)
    z2 = builder.add(z1, b)
    output = builder.relu(z2)

    # Build the graph
    print("\n3. Compiling graph...")
    graph = builder.build({"output": output})

    print(f"   Graph has {graph.operand_count} operands and {graph.operation_count} operations")
    print(f"   Inputs: {graph.get_input_names()}")
    print(f"   Outputs: {graph.get_output_names()}")

    # Execute original graph
    print("\n4. Executing original graph...")
    input_data = {
        "x": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    }

    results_original = context.compute(graph, input_data)
    print(f"   Input: {input_data['x']}")
    print(f"   Output: {results_original['output']}")

    # Save graph to file
    save_path = "example_graph.webnn"
    print(f"\n5. Saving graph to '{save_path}'...")
    graph.save(save_path)
    print(f"   Graph saved successfully")

    # Check file size
    file_size = os.path.getsize(save_path)
    print(f"   File size: {file_size} bytes")

    # Load graph from file
    print(f"\n6. Loading graph from '{save_path}'...")
    loaded_graph = webnn.MLGraph.load(save_path)

    print(f"   Loaded graph has {loaded_graph.operand_count} operands and {loaded_graph.operation_count} operations")
    print(f"   Inputs: {loaded_graph.get_input_names()}")
    print(f"   Outputs: {loaded_graph.get_output_names()}")

    # Execute loaded graph with same input
    print("\n7. Executing loaded graph with same input...")
    results_loaded = context.compute(loaded_graph, input_data)
    print(f"   Input: {input_data['x']}")
    print(f"   Output: {results_loaded['output']}")

    # Verify results match
    print("\n8. Verifying results match...")
    if np.allclose(results_original['output'], results_loaded['output']):
        print("   SUCCESS: Original and loaded graphs produce identical results!")
    else:
        print("   ERROR: Results differ!")
        print(f"   Difference: {np.abs(results_original['output'] - results_loaded['output'])}")

    # Try different input
    print("\n9. Testing loaded graph with different input...")
    test_input = {
        "x": np.array([[-1.0, 0.5, 2.0, -0.8]], dtype=np.float32)
    }

    results_test = context.compute(loaded_graph, test_input)
    print(f"   Input: {test_input['x']}")
    print(f"   Output: {results_test['output']}")

    # Display the saved JSON structure
    print("\n10. Saved graph structure (first 500 chars):")
    print("-" * 50)
    with open(save_path, 'r') as f:
        content = f.read()
        print(content[:500])
        if len(content) > 500:
            print(f"... ({len(content) - 500} more characters)")

    # Cleanup
    print(f"\n11. Cleaning up...")
    os.remove(save_path)
    print(f"   Removed '{save_path}'")

    print("\n" + "=" * 50)
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
