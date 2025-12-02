# train.py

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import LegalSENN
from data import (
    LegalTextDataset,
    build_label_map,
    save_label_map,
    load_label_map,
)


def get_device():
    """
    Prefer GPU if available.
    On your M3 Mac, this will use 'mps' (Metal Performance Shaders).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Train", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train LegalSENN on LexGLUE (SCOTUS/EURLEX/etc.)")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="outputs/legal_senn")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_concepts", type=int, default=16)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--text_col", type=str, default="text")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build or load label map
    label_map_path = os.path.join(args.output_dir, "label_map.json")
    if os.path.exists(label_map_path):
        label2id = load_label_map(label_map_path)
    else:
        label2id = build_label_map(args.train_csv, label_col=args.label_col)
        save_label_map(label2id, label_map_path)

    num_labels = len(label2id)

    # Datasets
    train_dataset = LegalTextDataset(
        csv_path=args.train_csv,
        tokenizer_name=args.encoder_name,
        label2id=label2id,
        max_length=args.max_length,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    val_dataset = LegalTextDataset(
        csv_path=args.val_csv,
        tokenizer_name=args.encoder_name,
        label2id=label2id,
        max_length=args.max_length,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    device = get_device()
    print(f"Using device: {device}")
    # Optional for M3: allow fallback
    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    model = LegalSENN(
        encoder_name=args.encoder_name,
        num_labels=num_labels,
        num_concepts=args.num_concepts,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    # Save config
    config = {
        "encoder_name": args.encoder_name,
        "num_labels": num_labels,
        "num_concepts": args.num_concepts,
        "max_length": args.max_length,
        "label_col": args.label_col,
        "text_col": args.text_col,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device
        )
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model to {ckpt_path} (val_acc={val_acc:.4f})")


if __name__ == "__main__":
    main()
