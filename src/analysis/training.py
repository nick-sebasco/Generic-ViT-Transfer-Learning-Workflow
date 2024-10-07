import torch
import torch.optim as optim
import ignite
from ignite.engine import Engine, Events
from ignite.metrics import Average, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint, ParamScheduler
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from typing import Tuple, Callable, Optional
from src.analysis.dataset import ZarrDataset, load_features
from src.analysis.models import MLP


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
) -> Callable[[ignite.engine.Engine, Tuple[torch.Tensor, torch.Tensor]], float]:
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
        return loss.item()
    return _validation_step


# Training Function
def run_training(
    zarr_path: str,
    resolution: str,
    hidden_dim: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    targets: Optional[torch.Tensor] = None,
    early_stopping: dict = {"patience": 5},
    lr_scheduler: dict = {"milestones": [5, 10], "gamma": 0.1},
    train_val_split: float = 0.8
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
    hidden_dim : int
        The size of the hidden layer in the MLP model.
    num_epochs : int
        The number of epochs to train for.
    batch_size : int
        The size of the mini-batches during training.
    learning_rate : float
        The learning rate for the Adam optimizer.
    targets: Optional[torch.Tensor] = None
        The target values.
    early_stopping : dict
        Dictionary containing parameters for early stopping (e.g., {"patience": 5}).
    lr_scheduler : dict
        Dictionary containing parameters for the learning rate scheduler (e.g., {"step_size": 5, "gamma": 0.1}).
    train_val_split: float = 0.8
        The fraction of data to be used for training and held out for validation.

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
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Utilize Ignite's ParamScheduler for learning rate decay using milestones
    lr_scheduler_ignite = ParamScheduler.param_scheduler.StepParamScheduler(
        optimizer=optimizer,
        param_name='lr',
        values=[learning_rate * (lr_scheduler["gamma"] ** i) for i in range(len(lr_scheduler["milestones"]) + 1)],
        milestones=lr_scheduler["milestones"]
    )

    # Create trainer and evaluator engines
    trainer = Engine(train_step(model, criterion, optimizer, device))
    evaluator = Engine(validation_step(model, criterion, device))

    # Attach the ParamScheduler to modify learning rate
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler_ignite)

    # Attach loss metrics to evaluator
    Loss(criterion).attach(evaluator, 'loss')

    # Define Tensorboard Logger
    tb_logger = TensorboardLogger(log_dir="./tb_logs")

    # Attach handler to log training loss at each iteration
    tb_logger.attach(
        trainer,
        log_handler=TensorboardLogger.OutputHandler(tag="training", output_transform=lambda loss: {"batch_loss": loss}),
        event_name=Events.ITERATION_COMPLETED,
    )

    # Attach handler to log validation loss at each epoch
    tb_logger.attach(
        evaluator,
        log_handler=TensorboardLogger.OutputHandler(tag="validation", metric_names=["loss"]),
        event_name=Events.EPOCH_COMPLETED,
    )

    # Early stopping
    early_stopping_handler = EarlyStopping(
        patience=early_stopping["patience"], 
        score_function=lambda engine: -engine.state.metrics['avg_loss'], 
        trainer=trainer
    )
    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    # Checkpointing
    checkpoint_handler = ModelCheckpoint(
        dirname='models', 
        filename_prefix='best', 
        n_saved=1,
        create_dir=True,
        score_function=lambda engine: -engine.state.metrics['avg_loss'], 
        score_name="val_loss"
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

    # Run training
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate_and_log_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Epoch {engine.state.epoch} - Validation Loss: {metrics['loss']:.4f}")

    trainer.run(train_loader, max_epochs=num_epochs)
    tb_logger.close()


# Usage
if __name__ == "__main__":
    run_training(
        zarr_path='path_to_feature_zarr',
        resolution='1',
        hidden_dim=512,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )