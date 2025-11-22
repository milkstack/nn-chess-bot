from __future__ import annotations
from typing import List, Optional, Union
import random
import numpy as np


class NeuralNet:
    def __init__(self, layer_config: List[int], parent: Optional['NeuralNet'] = None,
                 mutation_chance: float = 0.1, mutation_amount: float = 0.5):
        self.layer_config = layer_config
        self.layers = []
        self.mutation_chance = mutation_chance
        self.mutation_amount = mutation_amount

        if parent:
            # Deep copy layers from parent using NumPy for better performance, then mutate
            for parent_layer in parent.layers:
                new_layer = Layer(parent_layer.prev_layer_node_count, parent_layer.node_count)
                # Use NumPy arrays for faster copying
                if parent_layer._weights_np is not None:
                    new_layer._weights_np = parent_layer._weights_np.copy()
                    new_layer.weights = new_layer._weights_np.tolist()
                else:
                    new_layer.weights = [row[:] for row in parent_layer.weights]
                    new_layer._weights_np = None
                
                if parent_layer._biases_np is not None:
                    new_layer._biases_np = parent_layer._biases_np.copy()
                    new_layer.biases = new_layer._biases_np.tolist()
                else:
                    new_layer.biases = parent_layer.biases[:]
                    new_layer._biases_np = None
                
                new_layer.nodes_np = np.zeros(parent_layer.node_count, dtype=np.float32)
                self.layers.append(new_layer)
            
            for layer in self.layers:
                layer.mutate(mutation_chance=mutation_chance, mutation_amount=mutation_amount)

        else:
            for i in range(len(layer_config) - 1):
                layer = Layer(layer_config[i], layer_config[i + 1])
                layer.build_random()
                self.layers.append(layer)

    def evaluate(self, input: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Optimized evaluation using NumPy for vectorized operations.
        This significantly speeds up neural network inference.
        Passes numpy arrays directly between layers to avoid unnecessary conversions.
        
        Args:
            input: Input as list or numpy array
            
        Returns:
            Output as numpy array (no conversion to list for better performance)
        """
        # Convert input to numpy array once for faster computation
        if not isinstance(input, np.ndarray):
            result = np.array(input, dtype=np.float32)
        else:
            result = input.astype(np.float32) if input.dtype != np.float32 else input
        
        for i, layer in enumerate(self.layers):
            is_output_layer = (i == len(self.layers) - 1)
            # Pass numpy array directly (no conversion needed)
            result = layer.evaluate(result, is_output_layer=is_output_layer)

        return result
    
    def evaluate_batch(self, inputs: np.ndarray) -> np.ndarray:
        """
        Batch evaluate multiple inputs at once for better performance.
        This is much faster than calling evaluate() multiple times.
        
        Args:
            inputs: 2D numpy array of shape (batch_size, input_size)
            
        Returns:
            2D numpy array of shape (batch_size, output_size)
        """
        if inputs.ndim != 2:
            raise ValueError(f"inputs must be 2D array, got shape {inputs.shape}")
        
        # Ensure float32 for performance
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)
        
        result = inputs
        for i, layer in enumerate(self.layers):
            is_output_layer = (i == len(self.layers) - 1)
            # Evaluate batch through layer
            result = layer.evaluate_batch(result, is_output_layer=is_output_layer)
        
        return result

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
        """
        Initialize weights and biases using Xavier/Glorot initialization for better convergence.
        This prevents vanishing/exploding gradients by scaling weights based on layer size.
        """
        # Xavier/Glorot initialization: sqrt(6 / (fan_in + fan_out))
        # This ensures variance of activations stays roughly constant across layers
        limit = np.sqrt(6.0 / (self.prev_layer_node_count + self.node_count))
        
        # Generate random weights using Xavier initialization
        self._weights_np = np.random.uniform(-limit, limit, 
                                           (self.node_count, self.prev_layer_node_count)).astype(np.float32)
        # Biases initialized to small values (typically 0 or small random)
        self._biases_np = np.random.uniform(-0.1, 0.1, self.node_count).astype(np.float32)
        
        # Convert to lists for serialization compatibility
        self.weights = self._weights_np.tolist()
        self.biases = self._biases_np.tolist()

    def mutate(self, mutation_chance: float = 0.1, mutation_amount: float = 0.5):
        """
        Mutate weights and biases using vectorized NumPy operations for better performance.
        Updates both list and numpy versions.
        
        CRITICAL FIX: Added weight clipping to prevent unbounded weight growth over generations.
        This prevents saturation and loss of learning capacity.
        
        Args:
            mutation_chance: Probability that a weight or bias will be mutated (0.0 to 1.0)
            mutation_amount: Maximum absolute value for mutation amount. Mutations will be
                            in the range [-mutation_amount, +mutation_amount]
        """
        # Ensure numpy arrays exist
        if self._weights_np is None:
            self._weights_np = np.array(self.weights, dtype=np.float32)
        if self._biases_np is None:
            self._biases_np = np.array(self.biases, dtype=np.float32)
        
        # Vectorized mutation: generate random mask and mutations
        # For biases
        bias_mask = np.random.random(self.node_count) < mutation_chance
        bias_mutations = np.random.uniform(-mutation_amount, mutation_amount, self.node_count)
        self._biases_np += bias_mask * bias_mutations
        
        # For weights
        weight_mask = np.random.random((self.node_count, self.prev_layer_node_count)) < mutation_chance
        weight_mutations = np.random.uniform(-mutation_amount, mutation_amount, 
                                            (self.node_count, self.prev_layer_node_count))
        self._weights_np += weight_mask * weight_mutations
        
        # CRITICAL FIX: Clip weights and biases to prevent unbounded growth
        # This prevents saturation of activations and maintains learning capacity
        # Clipping to [-5, 5] is reasonable for ReLU networks
        self._weights_np = np.clip(self._weights_np, -5.0, 5.0)
        self._biases_np = np.clip(self._biases_np, -5.0, 5.0)
        
        # Update list versions
        self.weights = self._weights_np.tolist()
        self.biases = self._biases_np.tolist()

    def evaluate(self, inputArray: Union[List[float], np.ndarray], is_output_layer: bool = False) -> np.ndarray:
        """
        Optimized layer evaluation using NumPy vectorized operations.
        This replaces the nested loop with matrix multiplication for much faster computation.
        """
        # Convert input to numpy array if needed
        if not isinstance(inputArray, np.ndarray):
            inputArray = np.array(inputArray, dtype=np.float32)
        elif inputArray.dtype != np.float32:
            inputArray = inputArray.astype(np.float32)
        
        # Convert weights and biases to numpy arrays if they're not already cached
        if self._weights_np is None:
            self._weights_np = np.array(self.weights, dtype=np.float32)
        if self._biases_np is None:
            self._biases_np = np.array(self.biases, dtype=np.float32)
        
        # Vectorized matrix multiplication: nodes = weights @ input + biases
        # Using @ operator is faster than np.dot for 2D arrays
        self.nodes_np = self._weights_np @ inputArray + self._biases_np
        
        # Apply ReLU activation (skip for output layer)
        if not is_output_layer:
            self.nodes_np = np.maximum(self.nodes_np, 0.0)
        
        return self.nodes_np
    
    def evaluate_batch(self, inputs: np.ndarray, is_output_layer: bool = False) -> np.ndarray:
        """
        Batch evaluate multiple inputs at once.
        
        Args:
            inputs: 2D numpy array of shape (batch_size, input_size)
            is_output_layer: Whether this is the output layer
            
        Returns:
            2D numpy array of shape (batch_size, output_size)
        """
        if inputs.ndim != 2:
            raise ValueError(f"inputs must be 2D array, got shape {inputs.shape}")
        
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)
        
        # Ensure numpy arrays exist
        if self._weights_np is None:
            self._weights_np = np.array(self.weights, dtype=np.float32)
        if self._biases_np is None:
            self._biases_np = np.array(self.biases, dtype=np.float32)
        
        # Batch matrix multiplication: (batch_size, input_size) @ (input_size, output_size).T
        # Result: (batch_size, output_size)
        result = inputs @ self._weights_np.T + self._biases_np
        
        # Apply ReLU activation (skip for output layer)
        if not is_output_layer:
            result = np.maximum(result, 0.0)
        
        return result

