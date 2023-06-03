from datetime import datetime
import random
from pathlib import Path

from tqdm import tqdm
import hydra
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
from torch.utils.data import DataLoader
from PIL import Image
from utils import EarlyStopping
from logger import logger

from utils import SkyDataset, get_model
from evaluate import calculate_iou, calculate_mcc, calculate_pixel_accuracy


_DEEPLABV3_DIR = Path(__file__).absolute().parents[0]
experiment_base_dir = _DEEPLABV3_DIR / "experiments"
if not experiment_base_dir.exists():
    experiment_base_dir.mkdir()

np.random.seed(seed=42)
torch.manual_seed(seed=42)
random.seed(a=42)

interpolation_modes = {
    "NEAREST": Image.NEAREST,
    "BILINEAR": Image.BILINEAR,
    # Add other modes here if needed
}
optimize_metrics = ["accuracy", "val_loss", "miou", "mcc", "train_loss", None]


def train(
    model,
    criterion,
    optimizer,
    train_dataloader,
    max_epochs,
    device,
    val_dataloader=None,
    early_stop=None,
    monitored_metric=None,
    experiments_dir=None,
):
    logger.info("Starting Training ...")

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, torch.squeeze(labels, dim=1).long())
            loss.backward()
            optimizer.step()

            total_samples += labels.size(0)
            train_loss += loss * labels.size(0)

        mean_train_loss = train_loss / total_samples
        logger.info(f"Epoch [{epoch+1}/{max_epochs}]: Train: Loss: {mean_train_loss:.4f}")

        metrics = {"train_loss": mean_train_loss}

        if val_dataloader:
            # Evaluation on validation set
            model.eval()
            total_iou = 0.0
            total_accuracy = 0.0
            total_samples = 0
            val_loss = 0.0
            total_mcc = 0.0

            with torch.no_grad():
                for images, labels in tqdm(val_dataloader):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)["out"]
                    _, predicted = torch.max(outputs, 1)

                    loss = criterion(outputs, labels.squeeze(1).long())
                    val_loss += loss * labels.size(0)

                    iou = calculate_iou(predicted, labels)
                    accuracy = calculate_pixel_accuracy(
                        labels.cpu().numpy(), predicted.cpu().numpy()
                    )
                    mcc = calculate_mcc(labels.cpu().numpy(), predicted.cpu().numpy())

                    total_samples += labels.size(0)
                    total_iou += iou.item() * labels.size(0)
                    total_accuracy += accuracy * labels.size(0)
                    total_mcc += mcc * labels.size(0)

            metrics.update(
                {
                    "miou": total_iou / total_samples,
                    "accuracy": total_accuracy / total_samples,
                    "val_loss": val_loss / total_samples,
                    "mcc": total_mcc / total_samples,
                }
            )
            logger.info(
                f"Epoch [{epoch+1}/{max_epochs}]: Val: mIoU: {metrics['miou']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['val_loss']:.4f}, MCC: {metrics['mcc']:.4f}"
            )
        if monitored_metric not in metrics and early_stop:
            logger.warning(
                f"Provided unexpected metric to monitor: '{monitored_metric}'. If you are not using a validation dataset, disable early stopping or use 'train_loss'"
            )
            logger.error(
                f"Check 'optimize_metric' or disable early stopping! Aborting training..."
            )
            import sys

            sys.exit(1)

        if early_stop and not early_stop.improvement(
            model,
            metrics[monitored_metric],
            epoch,
        ):  # check for improvement and save model
            logger.info("Finished Training.\n")
            return early_stop.best_metric

    if not early_stop:
        EarlyStopping.save_model(model, f"model_last.pth", experiments_dir=experiments_dir)
        if monitored_metric in metrics:
            logger.info("Finished Training.\n")
            return metrics[monitored_metric]

    logger.info("Finished Training.\n")


@hydra.main(version_base=None, config_path=str(_DEEPLABV3_DIR), config_name="config")
def main(cfg):
    logger.info(f"Configuration: \n\n{OmegaConf.to_yaml(cfg)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert (
        cfg.optimize_metric in optimize_metrics
    ), f"Unexpected value for optimize_metric: {cfg.optimize_metric}"

    # create experiment directory
    cfg.experiment_name = (
        cfg.experiment_name if cfg.experiment_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    experiments_dir = experiment_base_dir / cfg.experiment_name
    if not experiments_dir.exists():
        experiments_dir.mkdir()
    OmegaConf.save(config=cfg, f=f"{str(experiments_dir)}/config_{cfg.experiment_name}.yaml")

    # transforms
    base_transform = (
        instantiate(
            cfg.base_transform,
            interpolation=interpolation_modes[cfg.base_transform.interpolation],
        )
        if cfg.base_transform
        else None
    )
    train_transform = instantiate(cfg.train_transform) if cfg.train_transform else None

    # train data  set up
    train_dataset = SkyDataset(
        cfg.data.train.image_path,
        cfg.data.train.label_path,
        train_transform=train_transform,
        base_transform=base_transform,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_params.batch_size,
        shuffle=True,
    )

    # val data set up
    if cfg.data.val.image_path:
        val_dataset = SkyDataset(
            cfg.data.val.image_path,
            cfg.data.val.label_path,
            base_transform=base_transform,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.train_params.batch_size,
        )
    else:
        logger.warning("Initiating training without validation set!")
        val_dataloader = None

    # model set up
    model = instantiate(cfg.model_params.model)
    model = get_model(
        model,
        device,
        checkpoint_path=cfg.model_params.checkpoint,
    )
    criterion = instantiate(cfg.criterion)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    early_stop = (
        instantiate(cfg.early_stopping, experiments_dir=experiments_dir)
        if cfg.train_params.enable_early_stopping
        else None
    )

    monitored_metric = train(
        model,
        criterion,
        optimizer,
        train_dataloader,
        cfg.train_params.max_epochs,
        device,
        val_dataloader=val_dataloader,
        early_stop=early_stop,
        monitored_metric=cfg.optimize_metric,
        experiments_dir=experiments_dir,
    )

    return monitored_metric


if __name__ == "__main__":
    main()
