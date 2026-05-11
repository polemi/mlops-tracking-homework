import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault('TORCH_HOME', str(Path('cache/torch').resolve()))


def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_csv(split_dir: Path) -> Path:
    candidates = [split_dir / "train.csv", split_dir / "test.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    csv_files = sorted(split_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV file was not found in {split_dir}")
    return csv_files[0]


def detect_image_column(df: pd.DataFrame) -> str:
    for column in ["file_name", "filename", "image", "image_path", "path"]:
        if column in df.columns:
            return column
    raise ValueError("Image path column was not found. Expected one of: file_name, filename, image, image_path, path")


def detect_label_column(df: pd.DataFrame) -> str:
    for column in ["label", "target", "class", "is_ai"]:
        if column in df.columns:
            return column
    raise ValueError("Label column was not found. Expected one of: label, target, class, is_ai")


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        split_dir: str,
        transform: Optional[transforms.Compose] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.csv_file = Path(csv_file)
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.data = pd.read_csv(self.csv_file)
        self.image_column = detect_image_column(self.data)
        self.label_column = detect_label_column(self.data)

        self.data[self.label_column] = self.data[self.label_column].astype(str)
        if label_to_idx is None:
            labels = sorted(self.data[self.label_column].unique().tolist())
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        unknown_labels = set(self.data[self.label_column].unique()) - set(self.label_to_idx)
        if unknown_labels:
            raise ValueError(f"Unknown labels in {self.csv_file}: {sorted(unknown_labels)}")

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_image_path(self, raw_path: str) -> Path:
        raw_path = str(raw_path)
        candidates = [self.split_dir / raw_path, self.csv_file.parent / raw_path, Path(raw_path)]
        if raw_path.startswith("train_data/"):
            alt_path = raw_path.replace("train_data/", "test_data/", 1)
            candidates.extend([self.split_dir / alt_path, self.csv_file.parent / alt_path])
        elif raw_path.startswith("test_data/"):
            alt_path = raw_path.replace("test_data/", "train_data/", 1)
            candidates.extend([self.split_dir / alt_path, self.csv_file.parent / alt_path])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image file was not found: {raw_path}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        image_path = self._resolve_image_path(row[self.image_column])
        image = Image.open(image_path).convert("RGB")
        label = self.label_to_idx[str(row[self.label_column])]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_dataloaders(
    data_dir: str,
    train_split: str,
    test_split: str,
    batch_size: int,
    num_workers: int,
    label_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    data_path = Path(data_dir)
    train_dir = data_path / train_split
    test_dir = data_path / test_split
    train_csv = find_csv(train_dir)
    test_csv = find_csv(test_dir)

    train_dataset = ImageClassificationDataset(
        csv_file=str(train_csv),
        split_dir=str(train_dir),
        transform=build_transforms(train=True),
        label_to_idx=label_to_idx,
    )
    test_dataset = ImageClassificationDataset(
        csv_file=str(test_csv),
        split_dir=str(test_dir),
        transform=build_transforms(train=False),
        label_to_idx=train_dataset.label_to_idx,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, train_dataset.label_to_idx


def build_model(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=pretrained)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_epoch(model, dataloader, criterion, device, optimizer=None, description="Train") -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(dataloader, desc=description):
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = classification_metrics(all_labels, all_preds)
    metrics["loss"] = float(total_loss / len(dataloader.dataset))
    return metrics


def save_checkpoint(model, label_to_idx: Dict[str, int], params: dict, metrics: dict, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "params": params,
            "metrics": metrics,
        },
        path_obj,
    )


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def create_s3_client(s3_cfg: dict):
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=s3_cfg["endpoint_url"],
        aws_access_key_id=s3_cfg["access_key"],
        aws_secret_access_key=s3_cfg["secret_key"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)


def upload_to_s3(local_path: str, s3_cfg: dict, key: str) -> str:
    client = create_s3_client(s3_cfg)
    ensure_bucket(client, s3_cfg["bucket"])
    client.upload_file(local_path, s3_cfg["bucket"], key)
    return f"s3://{s3_cfg['bucket']}/{key}"


def download_from_s3(local_path: str, s3_cfg: dict, key: str) -> str:
    client = create_s3_client(s3_cfg)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client.download_file(s3_cfg["bucket"], key, local_path)
    return local_path
