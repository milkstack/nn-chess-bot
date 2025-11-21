#!/usr/bin/env python3
"""Quick test script for 2-3-2 neural network"""

import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'bot'))

# Import directly from the module file
import neural_net
NeuralNet = neural_net.NeuralNet

def main():
    
    net = NeuralNet([64, 128, 128, 1])
    
    print("\nNetwork structure:")
    print(f"Layer config: {net.layer_config}")
    print(f"Number of layers: {len(net.layers)}")
    for i, layer in enumerate(net.layers):
        print(f"  Layer {i}: {layer.prev_layer_node_count} -> {layer.node_count} nodes")
        print(f"    Weights shape: {len(layer.weights)}x{len(layer.weights[0]) if layer.weights else 0}")
        print(f"    Biases: {layer.biases}")
    
    # Test with a simple input
    print("\n" + "="*50)
    print("Testing with input [1.0, 0.5]")
    print("="*50)
    input_data = [1.0, 0.5]
    output = net.evaluate(input_data)
    
    print(f"\nFinal output: {output}")
    print(f"Output length: {len(output)} (expected: 1)")

if __name__ == "__main__":
    main()

