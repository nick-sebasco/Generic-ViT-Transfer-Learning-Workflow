import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Callable


class MLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron (MLP) model that allows the user to specify 
    the number of hidden layers, the number of units in each layer, and the activation function.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input features.
    hidden_dims : List[int]
        A list of integers where each element specifies the number of units in the corresponding hidden layer.
    output_dim : int, optional
        The number of output units (default is 1, typically used for regression).
    activation : Callable, optional
        The activation function to apply after each hidden layer (default is ReLU).
    dropout_rate : float, optional
        The dropout rate for regularization (default is 0.0, which means no dropout).

    Attributes
    ----------
    mlp : nn.Sequential
        A sequential container holding the layers of the MLP.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        activation: Callable = nn.ReLU,
        dropout_rate: float = 0.0
    ):
        super(MLP, self).__init__()

        layers = []
        # Input layer to first hidden layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass of the flexible MLP.

        Parameters
        ----------
        x : Tensor
            A tensor of shape (batch_size, input_dim) representing the input data.

        Returns
        -------
        Tensor
            The output of the network, of shape (batch_size, output_dim).
        """
        return self.mlp(x)