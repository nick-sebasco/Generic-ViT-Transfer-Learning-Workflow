import argparse
from src.analysis.dataset import ZarrDataset
from src.analysis.training import run_training


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train the MLP head on extracted features.')

    # Positional arguments (as per the Nextflow script)
    parser.add_argument('training_image_ids', type=str,
                        help='Comma-separated list of training image IDs.')
    parser.add_argument('validation_image_ids', type=str,
                        help='Comma-separated list of validation image IDs.')
    parser.add_argument('feature_dir_path', type=str,
                        help='Path to the directory containing feature zarr files.')
    parser.add_argument('agg_type', type=str,
                        help='Aggregation type used in feature extraction (e.g., "mean").')
    parser.add_argument('resolution', type=str,
                        help='Resolution level used in feature extraction.')
    parser.add_argument('meta_csv', type=str,
                        help='Path to the metadata CSV file.')
    parser.add_argument('target_column', type=str,
                        help='Name of the target column in the metadata CSV.')
    parser.add_argument('class_order', type=str,
                        help='Comma-separated list of class labels for encoding (e.g., "0,1,2").')
    parser.add_argument('ordinal', type=str,
                        help='Boolean flag ("True" or "False") to use ordinal encoding.')

    # Optional arguments
    parser.add_argument('--hidden_dims', type=str, default='128,64',
                        help='Comma-separated list of hidden layer dimensions (e.g., "128,64").')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--output_dim', type=int, default=None,
                        help='Dimension of the model output. If not provided, it will be inferred.')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use TensorBoard for logging if this flag is set.')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Patience for early stopping.')
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                        help='Directory to save model checkpoints.')
    parser.add_argument('--checkpoint_backup_dir', type=str, default=None,
                        help='Backup directory for model checkpoints.')
    parser.add_argument('--tensorboard_log_dir', type=str, default='./tb_logs',
                        help='Directory for TensorBoard logs.')

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    training_image_ids = args.training_image_ids.split(',')
    validation_image_ids = args.validation_image_ids.split(',')

    class_order = args.class_order.split(',') if args.class_order else None
    if class_order is not None:
        # Try converting class_order to integers
        try:
            class_order = [int(cls) for cls in class_order]
        except ValueError:
            # Keep as strings if not integers
            pass

    hidden_dims = [int(hd) for hd in args.hidden_dims.split(',')]

    ordinal = args.ordinal.lower() == 'true'

    train_dataset = ZarrDataset(
        image_ids=training_image_ids,
        feature_zarr_dir=args.feature_dir_path,
        agg_type=args.agg_type,
        resolution=args.resolution,
        metadata_path=args.meta_csv,
        target_column=args.target_column,
        class_order=class_order,
        ordinal=ordinal,
    )

    val_dataset = ZarrDataset(
        image_ids=validation_image_ids,
        feature_zarr_dir=args.feature_dir_path,
        agg_type=args.agg_type,
        resolution=args.resolution,
        metadata_path=args.meta_csv,
        target_column=args.target_column,
        class_order=class_order,
        ordinal=ordinal,
    )

    if args.output_dim is not None:
        output_dim = args.output_dim
    elif class_order is not None:
        output_dim = len(class_order)
    else:
        output_dim = 1

    early_stopping = {
        "patience": args.early_stopping_patience,
        "score_function": lambda engine: -engine.state.metrics['loss'],
    }

    checkpointing = {
        "dirname": args.checkpoint_dir,
        "backup_location": args.checkpoint_backup_dir,
        "filename_prefix": "best",
        "n_saved": 2,
        "score_function": lambda engine: -engine.state.metrics['loss'],
        "score_name": "val_loss",
    }

    lr_scheduler = {
        "use_lr_scheduler": False,
        "milestone_values": [],
    }

    run_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        hidden_dims=hidden_dims,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dim=output_dim,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        checkpointing=checkpointing,
        use_tensorboard=args.use_tensorboard,
    )
