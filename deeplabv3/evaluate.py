import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader

from utils import SkyDataset, get_model
from logger import logger

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

_DEEPLABV3_DIR = Path(__file__).absolute().parents[0]

interpolation_modes = {
    "NEAREST": Image.NEAREST,
    "BILINEAR": Image.BILINEAR,
    # Add other modes here if needed
}


def calculate_iou(pred, target):
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    iou = intersection.float() / union.float()
    return iou


def calculate_pixel_accuracy(labels, predicted):
    total_pixels = labels.size
    correct_pixels = np.sum(labels == predicted)
    accuracy = correct_pixels / total_pixels
    return accuracy


def calculate_mcc(labels, predicted):
    mcc = matthews_corrcoef(labels.ravel(), predicted.ravel())
    return mcc


@hydra.main(version_base=None, config_path=str(_DEEPLABV3_DIR), config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset set up
    base_transform = (
        instantiate(
            cfg.base_transform,
            interpolation=interpolation_modes[cfg.base_transform.interpolation],
        )
        if cfg.base_transform
        else None
    )
    test_dataset = SkyDataset(
        cfg.evaluate.image_path,
        cfg.evaluate.label_path,
        base_transform=base_transform,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.evaluate.batch_size)

    # model set up
    model = instantiate(cfg.model_params.model)
    model = get_model(
        model,
        device,
        checkpoint_path=cfg.evaluate.model_path,
    )
    model.to(device)
    logger.info("Evaluating...\n")
    # Evaluation on test set
    model.eval()
    total_iou = 0.0
    total_accuracy = 0.0
    total_samples = 0
    total_mcc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)["out"]
            _, predicted = torch.max(outputs, 1)

            iou = calculate_iou(predicted, labels)
            accuracy = calculate_pixel_accuracy(labels.cpu().numpy(), predicted.cpu().numpy())
            mcc = calculate_mcc(labels.cpu().numpy(), predicted.cpu().numpy())

            total_samples += labels.size(0)
            total_iou += iou.item() * labels.size(0)
            total_accuracy += accuracy * labels.size(0)
            total_mcc += mcc * labels.size(0)

        mean_iou = total_iou / total_samples
        mean_accuracy = total_accuracy / total_samples
        mean_mcc = total_mcc / total_samples

        logger.info(f"IoU: {mean_iou:.4f}, Accuracy: {mean_accuracy:.4f}, MCC: {mean_mcc:.4f}")


if __name__ == "__main__":
    main()
