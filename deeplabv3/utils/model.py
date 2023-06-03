from pathlib import Path
from typing import Optional

import torch
from logger import logger


class EarlyStopping:
    """
    EarlyStopping can be used to stop the training if no improvement is observed for a certain number of epochs.
    """

    def __init__(self, patience: int = 5, delta: float = 0, experiments_dir: Path = None):
        """
        Initializes the EarlyStopping instance.

        Args:
            patience (int, optional): Number of epochs to wait for an improvement before stopping the training. Defaults to 5.
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
            experiments_dir (Path, optional): Current experiment directory to save models
        """
        self.patience = patience
        self.delta = delta
        self.best_metric = 0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.experiments_dir = experiments_dir

    def improvement(self, model: torch.nn.Module, metric: float, epoch: int) -> bool:
        """
        Checks if the metric has improved and saves the model if it has.

        Args:
            model (torch.nn.Module): The model to be saved.
            metric (float): The metric to be monitored.
            epoch (int): The current epoch number.

        Returns:
            bool: False if training should be stopped, True otherwise.
        """
        if metric > self.best_metric + self.delta:
            self.best_metric = metric
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            EarlyStopping.save_model(
                model,
                "model_best.pth",
                self.experiments_dir,
            )
            return True
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} with best metric value {self.best_metric} at epoch {self.best_epoch}..."
                )

                EarlyStopping.save_model(
                    model,
                    "model_last.pth",
                    self.experiments_dir,
                )
                return False
            else:
                return True

    @staticmethod
    def save_model(model: torch.nn.Module, filename: str, experiments_dir: Path = None):
        """
        Saves the model state to a file.

        Args:
            model (torch.nn.Module): The model to be saved.
            filename (str): The path to the file where the model state should be saved.
        """
        if experiments_dir:
            model_dir = experiments_dir / "models"
            if not model_dir.exists():
                model_dir.mkdir()

            filename = str(model_dir / filename)

        torch.save(model.state_dict(), filename)


def get_model(
    model: torch.nn.Module,
    device: str,
    checkpoint_path: Optional[str] = None,
) -> torch.nn.Module:
    """
    Loads a checkpoint to the model if avaialble else returns the model on device.

    Args:
        model (torch.nn.module): The initialized DeepLabV3 model.
        device (str): The device to which the model will be loaded.
                      This should be either 'cpu' or 'cuda'.
        checkpoint_path (str, optional): The path to a checkpoint file from which
                                         the model weights will be loaded. If this is
                                         None, the model will be initialized with
                                         random weights. Defaults to None.

    Returns:
        model (torch.nn.Module): The initialized DeepLabV3 model.
    """
    logger.info(f"Intialised model...")

    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
            )
        )
    return model.to(device)
