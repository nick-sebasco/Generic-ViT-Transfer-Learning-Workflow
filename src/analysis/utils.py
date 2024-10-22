import torch.optim as optim
import ignite
from ignite.handlers import PiecewiseLinear


def piecewise_linear_lr_scheduler(
    num_training_steps: int,
    optimizer: optim.Optimizer,
    milestone_values = [(0, 1e-5), (0.1, 5e-4), (1, 1e-6)]
):
    """
    Creates a PiecewiseLinear learning rate scheduler for the optimizer. The learning rate is updated at specified 
    milestones during training in a piecewise linear fashion.

    Parameters
    ----------
    num_training_steps : int
        The total number of training steps (typically num_epochs * len(dataloader)).
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be scheduled.
    milestone_values : list of tuple, optional
        A list of tuples specifying the fraction of training steps (0.0 to 1.0) and the corresponding learning rate.
        Each tuple is in the format (fraction_of_total_steps, learning_rate_value).

    Returns
    -------
    PiecewiseLinear
        A PiecewiseLinear scheduler handler for Ignite.

    Notes
    -----
    The learning rate will change according to the linear interpolation between the provided milestone values. For 
    example, if `milestone_values=[(0, 1e-5), (0.1, 5e-4), (1, 1e-6)]`, the learning rate will start at 1e-5, increase 
    to 5e-4 at 10% of training steps, and linearly decrease to 1e-6 by the end of training.
    """
    milestones_values = [(int(i * num_training_steps), j) for i, j in milestone_values]
    return PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)