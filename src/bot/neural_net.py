from __future__ import annotations
from typing import List, Optional, Union
import random
import numpy as np


class NeuralNet:
    def __init__(self, layer_config: List[int], parent: Optional['NeuralNet'] = None):
        self.layer_config = layer_config
        self.layers = []

        if parent:
            # Deep copy layers from parent, then mutate
            for parent_layer in parent.layers:
                new_layer = Layer(parent_layer.prev_layer_node_count, parent_layer.node_count)
                new_layer.weights = [row[:] for row in parent_layer.weights]  # Deep copy weights
                new_layer.biases = parent_layer.biases[:]  # Deep copy biases
                new_layer.nodes = [0.0] * parent_layer.node_count
                self.layers.append(new_layer)
            
            for layer in self.layers:
                layer.mutate()

        else:
            for i in range(len(layer_config) - 1):
                layer = Layer(layer_config[i], layer_config[i + 1])
                layer.build_random()
                self.layers.append(layer)

    def evaluate(self, input: List[float]) -> List[float]:
        """
        Optimized evaluation using NumPy for vectorized operations.
        This significantly speeds up neural network inference.
        Passes numpy arrays directly between layers to avoid unnecessary conversions.
        """
        # Convert input to numpy array once for faster computation
        result = np.array(input, dtype=np.float32)
        
        for i, layer in enumerate(self.layers):
            is_output_layer = (i == len(self.layers) - 1)
            # Pass numpy array directly (no conversion needed)
            result = layer.evaluate(result, is_output_layer=is_output_layer)

        return result.tolist()

    def get_state(self) -> dict:
        """
        Serialize the neural network state for multiprocessing.
        
        Returns:
            Dictionary containing layer configurations, weights, and biases
        """
        state = {
            "layer_config": self.layer_config,
            "layers": []
        }
        for layer in self.layers:
            state["layers"].append({
                "prev_layer_node_count": layer.prev_layer_node_count,
                "node_count": layer.node_count,
                "weights": [row[:] for row in layer.weights],  # Deep copy
                "biases": layer.biases[:]  # Deep copy
            })
        return state

    @staticmethod
    def from_state(state: dict) -> 'NeuralNet':
        """
        Reconstruct a neural network from serialized state.
        
        Args:
            state: Dictionary containing neural network state
            
        Returns:
            Reconstructed NeuralNet instance
        """
        net = NeuralNet.__new__(NeuralNet)  # Create without calling __init__
        net.layer_config = state["layer_config"]
        net.layers = []
        
        for layer_state in state["layers"]:
            layer = Layer(
                layer_state["prev_layer_node_count"],
                layer_state["node_count"]
            )
            layer.weights = [row[:] for row in layer_state["weights"]]
            layer.biases = layer_state["biases"][:]
            # Pre-initialize numpy arrays for faster evaluation (no lazy loading overhead)
            layer._weights_np = np.array(layer.weights, dtype=np.float32)
            layer._biases_np = np.array(layer.biases, dtype=np.float32)
            layer.nodes_np = np.zeros(layer_state["node_count"], dtype=np.float32)
            net.layers.append(layer)
        
        return net

    def get_model() -> object:
        return


class Layer: 
    def __init__(self, prev_layer_node_count: int, node_count: int):
        self.prev_layer_node_count = prev_layer_node_count
        self.node_count = node_count
        # Use numpy arrays from the start for better performance
        self.nodes_np = np.zeros(node_count, dtype=np.float32)
        self.weights = []  # Keep list format for serialization
        self.biases = []   # Keep list format for serialization
        self._weights_np = None  # Cached numpy array version
        self._biases_np = None   # Cached numpy array version

    def build_random(self):
        """Initialize weights and biases randomly using numpy for speed."""
        # Generate random weights and biases using numpy
        self._weights_np = np.random.uniform(-1, 1, (self.node_count, self.prev_layer_node_count)).astype(np.float32)
        self._biases_np = np.random.uniform(-1, 1, self.node_count).astype(np.float32)
        
        # Convert to lists for serialization compatibility
        self.weights = self._weights_np.tolist()
        self.biases = self._biases_np.tolist()

    def mutate(self):
        """Mutate weights and biases, updating both list and numpy versions."""
        # Mutate biases
        for i in range(self.node_count):
            if random.random() < 0.1:
                mutation_amount = random.uniform(-0.5, 0.5)
                self.biases[i] += mutation_amount
        
        # Mutate weights
        for i in range(self.node_count):
            for j in range(self.prev_layer_node_count):
                if random.random() < 0.1:
                    mutation_amount = random.uniform(-0.5, 0.5)
                    self.weights[i][j] += mutation_amount
        
        # Invalidate numpy cache so it gets regenerated on next evaluation
        self._weights_np = None
        self._biases_np = None

    def evaluate(self, inputArray: Union[List[float], np.ndarray], is_output_layer: bool = False) -> np.ndarray:
        """
        Optimized layer evaluation using NumPy vectorized operations.
        This replaces the nested loop with matrix multiplication for much faster computation.
        """
        # Convert input to numpy array if needed
        if not isinstance(inputArray, np.ndarray):
            inputArray = np.array(inputArray, dtype=np.float32)
        
        # Convert weights and biases to numpy arrays if they're not already cached
        if self._weights_np is None:
            self._weights_np = np.array(self.weights, dtype=np.float32)
        if self._biases_np is None:
            self._biases_np = np.array(self.biases, dtype=np.float32)
        
        # Vectorized matrix multiplication: nodes = weights @ input + biases
        # This is much faster than nested loops - O(n²) instead of O(n³) effectively
        self.nodes_np = np.dot(self._weights_np, inputArray) + self._biases_np
        
        # Apply ReLU activation (skip for output layer)
        if not is_output_layer:
            self.nodes_np = np.maximum(self.nodes_np, 0.0)
        
        return self.nodes_np

