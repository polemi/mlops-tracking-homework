import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mlops_utils import (
    build_dataloaders,
    build_model,
    download_from_s3,
    get_device,
    load_checkpoint,
    read_yaml,
    run_epoch,
    save_checkpoint,
    save_json,
    set_seed,
    upload_to_s3,
)


def log_hparams(writer: SummaryWriter, stage: str, params: dict) -> None:
    for key, value in params.items():
        if isinstance(value, (int, float, str, bool)):
            writer.add_text(f"{stage}/params/{key}", str(value), 0)


def log_metrics(writer: SummaryWriter, prefix: str, metrics: dict, step: int) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", value, step)


def train_stage(stage: str, params_path: str) -> dict:
    cfg = read_yaml(params_path)
    project_cfg = cfg["project"]
    model_cfg = cfg["model"]
    stage_cfg = cfg["train_base"] if stage == "base" else cfg["finetune"]
    s3_cfg = cfg["s3"]

    set_seed(stage_cfg["seed"])
    device = get_device()

    artifacts_dir = Path(project_cfg["artifacts_dir"])
    reports_dir = Path(project_cfg["reports_dir"])
    logs_dir = Path(project_cfg["logs_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_input_path = None
    label_to_idx = None
    if stage == "finetune":
        checkpoint_input_path = artifacts_dir / "base_model.pt"
        if s3_cfg.get("enabled", False):
            checkpoint_input_path = artifacts_dir / "base_model_from_s3.pt"
            download_from_s3(str(checkpoint_input_path), s3_cfg, s3_cfg["base_model_key"])
        checkpoint = load_checkpoint(str(checkpoint_input_path), device)
        label_to_idx = checkpoint["label_to_idx"]

    train_loader, test_loader, label_to_idx = build_dataloaders(
        data_dir=project_cfg["data_dir"],
        train_split=stage_cfg["train_split"],
        test_split=stage_cfg["test_split"],
        batch_size=stage_cfg["batch_size"],
        num_workers=stage_cfg["num_workers"],
        label_to_idx=label_to_idx,
    )

    model = build_model(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        freeze_backbone=model_cfg["freeze_backbone"],
    )

    if stage == "finetune":
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=stage_cfg["lr"],
        weight_decay=stage_cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=stage_cfg["step_size"],
        gamma=stage_cfg["gamma"],
    )

    writer = SummaryWriter(log_dir=str(logs_dir / stage))
    log_hparams(writer, stage, stage_cfg)
    log_hparams(writer, stage, model_cfg)

    history = []
    for epoch in range(1, stage_cfg["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, description=f"{stage} train {epoch}")
        scheduler.step()
        test_metrics = run_epoch(model, test_loader, criterion, device, optimizer=None, description=f"{stage} test {epoch}")

        log_metrics(writer, f"{stage}/train", train_metrics, epoch)
        log_metrics(writer, f"{stage}/test", test_metrics, epoch)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"{stage} epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.4f}, train_f1={train_metrics['f1_macro']:.4f}, "
            f"test_loss={test_metrics['loss']:.4f}, test_acc={test_metrics['accuracy']:.4f}, test_f1={test_metrics['f1_macro']:.4f}"
        )

    final_metrics = {
        "stage": stage,
        "device": str(device),
        "train_split": stage_cfg["train_split"],
        "test_split": stage_cfg["test_split"],
        "label_to_idx": label_to_idx,
        "params": stage_cfg,
        "history": history,
        "final_train_metrics": history[-1]["train"],
        "final_test_metrics": history[-1]["test"],
    }

    model_path = artifacts_dir / ("base_model.pt" if stage == "base" else "finetuned_model.pt")
    save_checkpoint(model, label_to_idx, stage_cfg, final_metrics, str(model_path))

    if s3_cfg.get("enabled", False):
        s3_key = s3_cfg["base_model_key"] if stage == "base" else s3_cfg["finetuned_model_key"]
        final_metrics["s3_model_uri"] = upload_to_s3(str(model_path), s3_cfg, s3_key)

    metrics_path = reports_dir / ("base_metrics.json" if stage == "base" else "finetuned_metrics.json")
    save_json(final_metrics, str(metrics_path))
    writer.close()
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["base", "finetune"], required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    train_stage(stage=args.stage, params_path=args.params)


if __name__ == "__main__":
    main()
