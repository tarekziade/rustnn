"""Type stubs for webnn package"""

from typing import Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt

class ML:
    """Entry point for the WebNN API"""

    def __init__(self) -> None: ...

    def create_context(
        self,
        device_type: str = "cpu",
        power_preference: str = "default"
    ) -> MLContext:
        """
        Create a new ML context

        Args:
            device_type: Device type ("cpu", "gpu", or "npu")
            power_preference: Power preference ("default", "high-performance", or "low-power")

        Returns:
            A new MLContext instance
        """
        ...

class MLContext:
    """Execution context for neural network graphs"""

    @property
    def power_preference(self) -> str:
        """Get the power preference hint"""
        ...

    @property
    def accelerated(self) -> bool:
        """
        Check if GPU/NPU acceleration is available

        Returns:
            True if the platform can provide GPU or NPU resources

        Note:
            This indicates platform capability, not a guarantee of device allocation.
            The actual execution may still use CPU if needed.
        """
        ...

    def create_graph_builder(self) -> MLGraphBuilder:
        """Create a graph builder for constructing computational graphs"""
        ...

    def compute(
        self,
        graph: MLGraph,
        inputs: Dict[str, npt.ArrayLike],
        outputs: Optional[Dict[str, npt.ArrayLike]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Execute the graph with given inputs

        Args:
            graph: The compiled MLGraph to execute
            inputs: Dictionary mapping input names to numpy arrays
            outputs: Optional pre-allocated output arrays

        Returns:
            Dictionary mapping output names to result numpy arrays
        """
        ...

    def convert_to_onnx(self, graph: MLGraph, output_path: str) -> None:
        """
        Convert graph to ONNX format

        Args:
            graph: The MLGraph to convert
            output_path: Path to save the ONNX model
        """
        ...

    def convert_to_coreml(self, graph: MLGraph, output_path: str) -> None:
        """
        Convert graph to CoreML format (macOS only)

        Args:
            graph: The MLGraph to convert
            output_path: Path to save the CoreML model
        """
        ...

    def dispatch(
        self,
        graph: MLGraph,
        inputs: Dict[str, "MLTensor"],
        outputs: Dict[str, "MLTensor"]
    ) -> None:
        """
        Dispatch graph execution asynchronously with MLTensor inputs/outputs

        Following the W3C WebNN MLTensor Explainer:
        https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md

        This method queues the graph for execution and returns immediately.
        Results are written to output tensors and can be read later with read_tensor().

        Args:
            graph: The compiled MLGraph to execute
            inputs: Dictionary mapping input names to MLTensor objects
            outputs: Dictionary mapping output names to MLTensor objects

        Note:
            This is currently implemented as synchronous execution.
            True async execution will be added in future versions.
        """
        ...

    def create_tensor(
        self,
        shape: List[int],
        data_type: str,
        readable: bool = True,
        writable: bool = True,
        exportable_to_gpu: bool = False
    ) -> "MLTensor":
        """
        Create a tensor for explicit tensor management

        Following the W3C WebNN MLTensor Explainer:
        https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md

        Args:
            shape: Shape of the tensor
            data_type: Data type string (e.g., "float32")
            readable: If True, tensor data can be read back to CPU (default: True)
            writable: If True, tensor data can be written from CPU (default: True)
            exportable_to_gpu: If True, tensor can be used as GPU texture (default: False)

        Returns:
            A new MLTensor with the specified properties
        """
        ...

    def read_tensor(self, tensor: "MLTensor") -> np.ndarray:
        """
        Read data from a tensor into a numpy array

        Follows the W3C WebNN MLTensor Explainer timeline model.

        Args:
            tensor: The MLTensor to read from (must have readable=True)

        Returns:
            The tensor data as a numpy array

        Raises:
            RuntimeError: If tensor is not readable or has been destroyed
        """
        ...

    def write_tensor(self, tensor: "MLTensor", data: npt.ArrayLike) -> None:
        """
        Write data from a numpy array into a tensor

        Follows the W3C WebNN MLTensor Explainer timeline model.

        Args:
            tensor: The MLTensor to write to (must have writable=True)
            data: Numpy array or array-like data to write

        Raises:
            RuntimeError: If tensor is not writable or has been destroyed
            ValueError: If data shape doesn't match tensor shape
        """
        ...

    def op_support_limits(self) -> Dict[str, any]:
        """
        Get operation support limits for this context

        Returns a dictionary describing what operations and parameter types
        are supported by the backend implementation. This allows applications
        to query feature support and adapt their models accordingly.

        The returned dictionary has the following structure:
        - preferredInputLayout: str - Preferred input layout ("nchw" or "nhwc")
        - maxTensorByteLength: int - Maximum tensor size in bytes
        - input: Dict - Tensor limits for input operands
        - constant: Dict - Tensor limits for constant operands
        - output: Dict - Tensor limits for output operands
        - <operation_name>: Dict - Per-operation support limits

        Each tensor limits dict contains:
        - dataTypes: List[str] - Supported data types
        - rankRange: Dict - min/max supported ranks

        Returns:
            Dictionary with support limits for each operation

        Example:
            >>> limits = context.op_support_limits()
            >>> print(limits['preferredInputLayout'])
            'nchw'
            >>> print(limits['input']['dataTypes'])
            ['float32', 'float16', 'int32', 'uint32', 'int8', 'uint8', 'int64', 'uint64']
            >>> print(limits['relu']['input']['dataTypes'])
            ['float32', 'float16']
        """
        ...

class MLOperand:
    """Represents an operand in the computational graph"""

    @property
    def data_type(self) -> str:
        """Get the operand's data type"""
        ...

    @property
    def shape(self) -> List[int]:
        """Get the operand's shape"""
        ...

    @property
    def name(self) -> Optional[str]:
        """Get the operand's name"""
        ...

class MLGraphBuilder:
    """Builder for constructing WebNN computational graphs"""

    def __init__(self) -> None: ...

    def input(
        self,
        name: str,
        shape: List[int],
        data_type: str = "float32"
    ) -> MLOperand:
        """
        Create an input operand

        Args:
            name: Name of the input
            shape: List of dimensions
            data_type: Data type string (e.g., "float32")

        Returns:
            The created input operand
        """
        ...

    def constant(
        self,
        value: npt.ArrayLike,
        shape: Optional[List[int]] = None,
        data_type: Optional[str] = None
    ) -> MLOperand:
        """
        Create a constant operand from numpy array

        Args:
            value: NumPy array or Python list
            shape: Optional shape override
            data_type: Optional data type string

        Returns:
            The created constant operand
        """
        ...

    # Binary operations
    def add(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Element-wise addition"""
        ...

    def sub(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Element-wise subtraction"""
        ...

    def mul(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Element-wise multiplication"""
        ...

    def div(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Element-wise division"""
        ...

    def matmul(self, a: MLOperand, b: MLOperand) -> MLOperand:
        """Matrix multiplication"""
        ...

    # Unary operations
    def relu(self, x: MLOperand) -> MLOperand:
        """ReLU activation"""
        ...

    def sigmoid(self, x: MLOperand) -> MLOperand:
        """Sigmoid activation"""
        ...

    def tanh(self, x: MLOperand) -> MLOperand:
        """Tanh activation"""
        ...

    def softmax(self, x: MLOperand) -> MLOperand:
        """Softmax activation"""
        ...

    def reshape(self, x: MLOperand, new_shape: List[int]) -> MLOperand:
        """Reshape operation"""
        ...

    def build(self, outputs: Dict[str, MLOperand]) -> MLGraph:
        """
        Build and compile the computational graph

        Args:
            outputs: Dictionary mapping output names to MLOperand objects

        Returns:
            The compiled graph
        """
        ...

class MLGraph:
    """Compiled computational graph"""

    @property
    def operand_count(self) -> int:
        """Get the number of operands in the graph"""
        ...

    @property
    def operation_count(self) -> int:
        """Get the number of operations in the graph"""
        ...

    def get_input_names(self) -> List[str]:
        """Get list of input names"""
        ...

    def get_output_names(self) -> List[str]:
        """Get list of output names"""
        ...

    def save(self, path: str) -> None:
        """
        Save the graph to a .webnn JSON file

        Args:
            path: File path to save the graph (e.g., "model.webnn")

        Example:
            >>> graph.save("my_model.webnn")
        """
        ...

    @staticmethod
    def load(path: str) -> "MLGraph":
        """
        Load a graph from a .webnn JSON file

        Args:
            path: File path to load the graph from (e.g., "model.webnn")

        Returns:
            The loaded MLGraph

        Example:
            >>> graph = MLGraph.load("my_model.webnn")
        """
        ...

class MLTensor:
    """
    MLTensor represents an opaque typed tensor with data storage

    MLTensor is used for explicit tensor management in WebNN, allowing
    pre-allocation of input/output buffers and explicit data transfer.

    Following the W3C WebNN MLTensor Explainer:
    https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md
    """

    @property
    def data_type(self) -> str:
        """Get the data type of the tensor (e.g., 'float32', 'int32')"""
        ...

    @property
    def shape(self) -> List[int]:
        """Get the shape of the tensor"""
        ...

    @property
    def size(self) -> int:
        """Get the total number of elements in the tensor"""
        ...

    @property
    def readable(self) -> bool:
        """Check if tensor data can be read back to CPU"""
        ...

    @property
    def writable(self) -> bool:
        """Check if tensor data can be written from CPU"""
        ...

    @property
    def exportable_to_gpu(self) -> bool:
        """Check if tensor can be exported for use as GPU texture"""
        ...

    def destroy(self) -> None:
        """
        Destroy the tensor and release its resources

        After calling destroy(), the tensor cannot be used for any operations.
        This follows the W3C WebNN MLTensor Explainer for explicit resource management.

        Raises:
            RuntimeError: If tensor has already been destroyed
        """
        ...
