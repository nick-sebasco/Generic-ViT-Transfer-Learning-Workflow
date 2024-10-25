import os


class Config:
    """
    Default configurations with environment variable support.
    """
    CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', 'models')
    CHECKPOINT_BACKUP_DIR = os.getenv('CHECKPOINT_BACKUP_DIR', None)
    EPOCHS = int(os.getenv('EPOCHS', 10))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))
    USE_TENSORBOARD = os.getenv('USE_TENSORBOARD', 'False').lower() == 'true'