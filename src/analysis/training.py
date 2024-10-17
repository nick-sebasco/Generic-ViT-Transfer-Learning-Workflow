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
from src.analysis.dataset import ZarrDataset, load_features
from src.analysis.models import MLP
from src.analysis.utils import piecewise_linear_lr_scheduler


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
            y_hat = model(x)
            y_hat = torch.squeeze(y_hat)
            loss = criterion(y_hat, t.float())
        return y_hat, t  # Return predictions and targets
    return _validation_step


# Training Function
def run_training(
    zarr_path: str,
    resolution: str,
    hidden_dims: List[int],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dim: int = 1,
    # TODO: image id key in csv that corresponds to target information
    targets: Optional[torch.Tensor] = None,
    early_stopping: dict = {
        "patience": 5,
        "score_function": lambda engine: -engine.state.metrics['loss']
    },
    lr_scheduler: dict = {
        "use_lr_scheduler": True,
        "milestone_values": [(0, 1e-5), (0.1, 5e-4), (1, 1e-6)]
    },
    checkpointing: dict = {
        "dirname": "models",
        "backup_location": None, # secondary checkpoint save location.  ex. "backups"
        "filename_prefix": "best",
        "n_saved": 2,
        "score_function": lambda engine: -engine.state.metrics['loss'],
        "score_name": "val_loss"
    },
    train_val_split: float = 0.8,
    use_tensorboard: bool = True
) -> None:
    """
    Runs the training and validation loop using Ignite's Engine, handling dataset preparation, model initialization, 
    and optimization. It sets up the training and evaluation engines, attaches necessary handlers, and starts the training process.

    Parameters
    ----------
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
    # Load features and create dataset
    features = load_features(zarr_path, resolution)
    if targets is None:
        raise ValueError("targets must be provided by user.")

    dataset = ZarrDataset(features, targets)
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = features.shape[0]
    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create trainer and evaluator engines
    trainer = Engine(train_step(model, criterion, optimizer, device))
    evaluator = Engine(validation_step(model, criterion, device))

    # Attach learning rate scheduler
    if lr_scheduler["use_lr_scheduler"]:
        num_training_steps = num_epochs * len(train_loader)
        scheduler = piecewise_linear_lr_scheduler(
            num_training_steps,
            optimizer,
            lr_scheduler["milestone_values"]
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    # Attach loss metrics to evaluator
    Loss(criterion).attach(evaluator, 'loss')

    if use_tensorboard:
        # Define Tensorboard Logger
        tb_logger = TensorboardLogger(log_dir="./tb_logs")

        # Attach handler to log training loss at each iteration
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda loss: {"batch_loss": loss}
        )

        # Attach handler to log validation loss at each epoch
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=["loss"]
        )

    # Early stopping
    early_stopping_handler = EarlyStopping(
        patience=early_stopping["patience"], 
        score_function=early_stopping["score_function"], 
        trainer=trainer
    )
    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    # Run training
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate_and_log_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Epoch {engine.state.epoch} - Validation Loss: {metrics['loss']:.4f}")

    # Checkpointing
    checkpoint_handler = ModelCheckpoint(
        dirname=checkpointing["dirname"],
        filename_prefix=checkpointing["filename_prefix"],
        n_saved=checkpointing["n_saved"],
        create_dir=True,
        score_function=checkpointing["score_function"],
        score_name=checkpointing["score_name"],
        require_empty=False,  # Allow saving even if the directory is not empty
        global_step_transform=global_step_from_engine(trainer)
    )

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

    # save checkpoints to two places
    if checkpointing["backup_location"] is not None:
        checkpoint_handler_secondary = ModelCheckpoint(
            dirname=checkpointing["backup_location"],
            filename_prefix=checkpointing["filename_prefix"],
            n_saved=checkpointing["n_saved"],
            create_dir=True,
            score_function=checkpointing["score_function"],
            score_name=checkpointing["score_name"],
            require_empty=False,
            global_step_transform=global_step_from_engine(trainer)
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler_secondary, {"model": model})
    
    # Start trainer
    trainer.run(train_loader, max_epochs=num_epochs)

    if use_tensorboard:
        tb_logger.close()
