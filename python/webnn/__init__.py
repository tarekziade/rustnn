"""
WebNN Python API

This package provides Python bindings for the WebNN (Web Neural Network) API,
allowing you to build, validate, and execute neural network graphs.

Example usage:
    >>> import webnn
    >>> ml = webnn.ML()
    >>> context = ml.create_context(accelerated=False)
    >>> builder = context.create_graph_builder()
    >>>
    >>> # Build a simple graph
    >>> x = builder.input("x", [2, 3], "float32")
    >>> y = builder.input("y", [2, 3], "float32")
    >>> z = builder.add(x, y)
    >>> output = builder.relu(z)
    >>>
    >>> # Compile the graph
    >>> graph = builder.build({"output": output})
"""

import asyncio
from typing import Dict, Optional
import numpy as np

from ._rustnn import (
    ML,
    MLContext,
    MLGraphBuilder,
    MLOperand,
    MLGraph,
    MLTensor,
)


class AsyncMLContext:
    """Async wrapper for MLContext providing WebNN-compliant async execution.

    This class wraps the synchronous MLContext and provides asynchronous methods
    for non-blocking execution, following the W3C WebNN specification.

    Example:
        >>> ml = webnn.ML()
        >>> context = ml.create_context(accelerated=False)
        >>> async_context = webnn.AsyncMLContext(context)
        >>>
        >>> # Async execution
        >>> await async_context.dispatch(graph, inputs)
        >>> result = await async_context.read_tensor_async(output_tensor)
    """

    def __init__(self, context: MLContext):
        """Initialize async context wrapper.

        Args:
            context: Synchronous MLContext instance to wrap
        """
        self._context = context

    async def dispatch(
        self,
        graph: MLGraph,
        inputs: Dict[str, MLTensor],
        outputs: Dict[str, MLTensor]
    ) -> None:
        """Execute graph computation asynchronously (WebNN MLTensor Explainer).

        Following the W3C WebNN MLTensor Explainer:
        https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md

        This method dispatches the graph for execution and returns immediately.
        Results should be read using read_tensor_async() on output tensors.

        Args:
            graph: The compiled MLGraph to execute
            inputs: Dictionary mapping input names to MLTensor objects
            outputs: Dictionary mapping output names to MLTensor objects

        Returns:
            None (execution happens asynchronously)

        Example:
            >>> input_tensor = context.create_tensor([2, 3], "float32")
            >>> output_tensor = context.create_tensor([2, 3], "float32")
            >>> await async_context.dispatch(graph, {"x": input_tensor}, {"out": output_tensor})
        """
        # Run synchronous dispatch in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._context.dispatch, graph, inputs, outputs)

    async def read_tensor_async(self, tensor: MLTensor) -> np.ndarray:
        """Read tensor data asynchronously.

        Args:
            tensor: The MLTensor to read from

        Returns:
            numpy.ndarray: The tensor data

        Example:
            >>> result = await async_context.read_tensor_async(output_tensor)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._context.read_tensor, tensor)

    async def write_tensor_async(self, tensor: MLTensor, data: np.ndarray) -> None:
        """Write tensor data asynchronously.

        Args:
            tensor: The MLTensor to write to
            data: Numpy array data to write

        Example:
            >>> await async_context.write_tensor_async(input_tensor, data)
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._context.write_tensor, tensor, data)

    # Synchronous methods pass through to underlying context
    def create_graph_builder(self) -> MLGraphBuilder:
        """Create a graph builder (synchronous)."""
        return self._context.create_graph_builder()

    def create_tensor(self, shape, data_type: str, readable: bool = True,
                      writable: bool = True, exportable_to_gpu: bool = False) -> MLTensor:
        """Create a tensor (synchronous)."""
        return self._context.create_tensor(shape, data_type, readable, writable, exportable_to_gpu)

    def compute(self, graph: MLGraph, inputs: Dict[str, np.ndarray], outputs=None):
        """Compute (synchronous) - prefer dispatch() for async execution."""
        return self._context.compute(graph, inputs, outputs)

    def read_tensor(self, tensor: MLTensor) -> np.ndarray:
        """Read tensor (synchronous) - prefer read_tensor_async() for async."""
        return self._context.read_tensor(tensor)

    def write_tensor(self, tensor: MLTensor, data: np.ndarray) -> None:
        """Write tensor (synchronous) - prefer write_tensor_async() for async."""
        return self._context.write_tensor(tensor, data)

    @property
    def accelerated(self) -> bool:
        """Check if GPU/NPU acceleration is available."""
        return self._context.accelerated

    @property
    def power_preference(self) -> str:
        """Get power preference."""
        return self._context.power_preference


__all__ = [
    "ML",
    "MLContext",
    "AsyncMLContext",
    "MLGraphBuilder",
    "MLOperand",
    "MLGraph",
    "MLTensor",
]

__version__ = "0.1.0"
