import torch
import torch.optim as optim
import ignite
from ignite.engine import Engine, Events
from ignite.metrics import Average, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from ignite.handlers.tensorboard_logger import *
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from typing import Tuple, Callable, Optional, List
from .dataset import ZarrDataset, load_features
from .models import MLP
from .utils import piecewise_linear_lr_scheduler
import os


# Training Step Function
def train_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Callable[[ignite.engine.Engine, Tuple[torch.Tensor, torch.Tensor]], float]:
    """
    Returns a training step function to be used in an Ignite Engine during each iteration.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    criterion : torch.nn.Module
        The loss function used for training.
    optimizer : torch.optim.Optimizer
        The optimizer used to update model parameters.
    device : torch.device
        The device to perform training on, typically 'cuda' or 'cpu'.

    Returns
    -------
    Callable
        A function that takes in an Ignite engine and a batch (features and targets) 
        and performs one training iteration, returning the computed loss for the batch.
    """
    def _train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, t = batch
        x, t = x.to(device), t.to(device)
        t=torch.squeeze(t)
        y_hat = model(x)
        y_hat = torch.squeeze(y_hat)
        loss = criterion(y_hat, t.float())
        loss.backward()
        optimizer.step()
        return loss.item()
    return _train_step


# Validation Step Function
def validation_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device
) -> Callable[[ignite.engine.Engine, Tuple[torch.Tensor, torch.Tensor]], float]: # TODO: return value is off
    """
    Returns a validation step function to be used in an Ignite Engine for each iteration of validation.
    This function performs forward propagation on the model in evaluation mode and computes the loss.

    Parameters
    ----------
    model : torch.nn.Module
        The model being evaluated.
    criterion : torch.nn.Module
        The loss function used for validation.
    device : torch.device
        The device to perform validation on, typically 'cuda' or 'cpu'.

    Returns
    -------
    Callable
        A function that takes an Ignite engine and a batch (features and targets) and performs one validation iteration,
        returning the computed loss for the batch.
    """
    def _validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, t = batch
            x, t = x.to(device), t.to(device)
            t=torch.squeeze(t)
            y_hat = model(x)
            y_hat = torch.squeeze(y_hat)
            loss = criterion(y_hat, t.float())
        return y_hat, t  # Return predictions and targets
    return _validation_step


# Training Function
def run_training(
    model_name: str,
    train_dataset: ZarrDataset,
    val_dataset: ZarrDataset,
    hidden_dims: List[int],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    activation_function: torch.nn.Module,
    loss_function: torch.nn.modules.loss._Loss,
    model_out_dir: str = None,
    output_dim: int = 1,
    early_stopping: dict = {
        "patience": 5,
        "score_function": lambda engine: -engine.state.metrics['loss'],
    },
    lr_scheduler: dict = {
        "use_lr_scheduler": True,
        "milestone_values": [(0, 1e-5), (0.1, 5e-4), (1, 1e-6)],
    },
    checkpointing: dict = {
        "dirname": "models",
        "backup_location": None,  # secondary checkpoint save location. ex. "backups"
        "filename_prefix": "best",
        "n_saved": 2,
        "score_function": lambda engine: -engine.state.metrics['loss'],
        "score_name": "val_loss",
    },
    use_tensorboard: bool = True
) -> None:
    """
    Runs the training and validation loop using Ignite's Engine, handling dataset preparation, model initialization, 
    and optimization. It sets up the training and evaluation engines, attaches necessary handlers, and starts the training process.

    Parameters
    ----------
    model
    model_name : str
        prefix for the final model's torchscript file name.
    zarr_path : str
        The path to the Zarr file storing the feature data.
    resolution : str
        The resolution level to load from the Zarr dataset.
    hidden_dims : List[int]
        A list of integers where each element specifies the number of units in the corresponding hidden layer.
    num_epochs : int
        The number of epochs to train for.
    batch_size : int
        The size of the mini-batches during training.
    learning_rate : float(
        The learning rate for the Adam optimizer.
    model_out_dir : str = None
        directory to save the final model's torchscript file to outside of the NextFlow work directory if desired. 
    targets: Optional[torch.Tensor] = None
        The target values.
    early_stopping : dict
        Dictionary containing parameters for early stopping (e.g., {"patience": 5}).
    lr_scheduler: dict
        Dictionary containing parameters for the learning rate scheduler (e.g., {"step_size": 5, "gamma": 0.1}).
    checkpointing : dict
        Dictionary containing parameters for checkpointing (e.g., {"dirname": "models", "filename_prefix": "best"}).
    train_val_split: float = 0.8
        The fraction of data to be used for training and held out for validation.
    use_tensorboard: bool = True
        A boolean which determines whether or not to use tensorboard.

    Returns
    -------
    None
    """
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine input dimension from a sample
    sample_feature, _ = train_dataset[0]
    input_dim = sample_feature.numel()
    model = torch.nn.Sequential(MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim),activation_function).to(device)

    # Define loss and optimizer
    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create trainer and evaluator engines
    trainer = Engine(train_step(model, criterion, optimizer, device))
    evaluator = Engine(validation_step(model, criterion, device))

    # Attach learning rate scheduler
    if lr_scheduler["use_lr_scheduler"]:
        num_training_steps = num_epochs * len(train_loader)
        scheduler = piecewise_linear_lr_scheduler(
            num_training_steps, optimizer, lr_scheduler["milestone_values"]
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    # Attach loss metrics to evaluator
    Loss(criterion).attach(evaluator, 'loss')

    # TensorBoard logging
    if use_tensorboard:
        tb_logger = TensorboardLogger(log_dir="./tb_logs")
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda loss: {"batch_loss": loss},
        )
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=["loss"],
        )
    """
    # Early stopping
    early_stopping_handler = EarlyStopping(
        patience=early_stopping["patience"],
        score_function=early_stopping["score_function"],
        trainer=trainer,
    )
    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)
    """
    # Checkpointing
    checkpoint_handler = ModelCheckpoint(
        dirname=checkpointing["dirname"],
        filename_prefix=checkpointing["filename_prefix"],
        n_saved=checkpointing["n_saved"],
        create_dir=True,
        score_function=checkpointing["score_function"],
        score_name=checkpointing["score_name"],
        require_empty=False,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

    # Secondary checkpointing (if backup location is provided)
    if checkpointing.get("backup_location"):
        checkpoint_handler_secondary = ModelCheckpoint(
            dirname=checkpointing["backup_location"],
            filename_prefix=checkpointing["filename_prefix"],
            n_saved=checkpointing["n_saved"],
            create_dir=True,
            score_function=checkpointing["score_function"],
            score_name=checkpointing["score_name"],
            require_empty=False,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler_secondary, {"model": model}
        )

    # Validation and logging
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate_and_log_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Epoch {engine.state.epoch} - Validation Loss: {metrics['loss']:.4f}")

    # Start training
    trainer.run(train_loader, max_epochs=num_epochs)
    model_scripted = torch.jit.script(model)
    model_scripted.save(os.path.join("models",f"{model_name}_final.pt"))
    if model_out_dir is not None:
        model_scripted.save(os.path.join(model_out_dir,f"{model_name}_final.pt"))
    if use_tensorboard:
        tb_logger.close()
