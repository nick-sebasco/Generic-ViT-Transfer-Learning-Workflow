import pytest
import torch
import torch.nn as nn
from src.analysis.training import train_step, validation_step


def test_train_step(
    small_dummy_train_validate_data,
    simple_train_validate_model,
    simple_train_validate_optimizer,
    criterion
):
    """Test the train_step function with small dummy data."""
    device = 'cpu'
    inputs, targets = small_dummy_train_validate_data

    # Convert targets to float since BCEWithLogitsLoss expects floating-point targets
    targets = targets.float()

    # Get the actual _train_step function by calling train_step with arguments
    _train_step = train_step(simple_train_validate_model, criterion, simple_train_validate_optimizer, device)
    
    # Perform one training step, passing the batch as argument
    loss = _train_step(None, (inputs, targets))

    # Check the loss value is a positive scalar
    assert isinstance(loss, float)
    assert loss > 0

    # Ensure the model's weights have been updated by checking that the gradients are not None
    for param in simple_train_validate_model.parameters():
        assert param.grad is not None


def test_validation_step(
    small_dummy_train_validate_data,
    simple_train_validate_model,
    criterion
):
    """Test the validation_step function with small dummy data."""
    device = 'cpu'
    inputs, targets = small_dummy_train_validate_data

    # Convert targets to float since BCEWithLogitsLoss expects floating-point targets
    targets = targets.float()

    # Get the actual _validation_step function by calling validation_step with arguments
    _validation_step = validation_step(simple_train_validate_model, criterion, device)
    
    # Perform one validation step, passing the batch as argument
    loss = _validation_step(None, (inputs, targets))

    # Check the loss value is a positive scalar
    assert isinstance(loss, float)
    assert loss > 0

    # Ensure the model's parameters have not been updated (no gradient accumulation during validation)
    for param in simple_train_validate_model.parameters():
        assert param.grad is None