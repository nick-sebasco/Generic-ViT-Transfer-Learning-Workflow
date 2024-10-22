import pytest
import torch
import torch.nn as nn
from src.analysis.models import MLP


def test_mlp_initialization():
    """Test initialization of the MLP model."""
    model = MLP(input_dim=512, hidden_dims=[256, 128], output_dim=10, activation=nn.ReLU)
    
    # Ensure the model has the correct number of layers
    layers = list(model.mlp)
    assert len(layers) == 5  # 2 hidden layers, each followed by ReLU, and 1 output layer
    
    # Check that the layers are of the expected type
    assert isinstance(layers[0], nn.Linear)
    assert isinstance(layers[1], nn.ReLU)
    assert isinstance(layers[2], nn.Linear)
    assert isinstance(layers[3], nn.ReLU)
    assert isinstance(layers[4], nn.Linear)


def test_mlp_forward(dummy_input):
    """Test forward pass of the MLP model."""
    model = MLP(input_dim=512, hidden_dims=[256, 128], output_dim=10)
    output = model(dummy_input)
    
    # Ensure output has the correct shape
    assert output.shape == (32, 10)


def test_mlp_with_dropout(dummy_input):
    """Test MLP with dropout layers."""
    model = MLP(input_dim=512, hidden_dims=[256, 128], output_dim=10, dropout_rate=0.5)
    
    # Ensure dropout layers are present
    layers = list(model.mlp)
    assert isinstance(layers[2], nn.Dropout)
    assert isinstance(layers[5], nn.Dropout)

    # Test forward pass with dropout
    output = model(dummy_input)
    assert output.shape == (32, 10)


def test_mlp_different_activation(dummy_input):
    """Test MLP with a different activation function."""
    model = MLP(input_dim=512, hidden_dims=[256, 128], output_dim=10, activation=nn.Tanh)
    
    # Ensure Tanh is used instead of ReLU
    layers = list(model.mlp)
    assert isinstance(layers[1], nn.Tanh)
    assert isinstance(layers[3], nn.Tanh)

    output = model(dummy_input)
    assert output.shape == (32, 10)

