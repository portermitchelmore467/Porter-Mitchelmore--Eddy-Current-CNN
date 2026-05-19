# train_model.py (training script for the CNN model)

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os



# Reproducibility helper
def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# import for saving training history
import json

# Add project root to sys.path!!!!
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

from data.impedance_dataset import build_transforms
from models.crack_classifier import build_model


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"loss": running_loss / total, "acc": correct / total})
    return running_loss / total, correct / total


def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": running_loss / total, "acc": correct / total})
    return running_loss / total, correct / total


# Grouped Train/Val/Test Split (Group by Run)

def grouped_split_by_run(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    set_seed(seed)
    # Extract run ID
    df["run"] = df["image_path"].apply(
        lambda x: int(x.split("_run")[1].split("_")[0])
    )

    unique_runs = sorted(df["run"].unique())
    n_runs = len(unique_runs)

    # Compute number of runs per split
    n_train = max(1, int(n_runs * train_ratio))
    n_val = max(1, int(n_runs * val_ratio))
    n_test = n_runs - n_train - n_val

    # Fix rounding issues
    if n_test < 1:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    # Shuffle runs
    import random
    random.shuffle(unique_runs)

    train_runs = unique_runs[:n_train]
    val_runs = unique_runs[n_train:n_train + n_val]
    test_runs = unique_runs[n_train + n_val:]

    # Debug print to verify no leakage
    print("Train runs:", train_runs)
    print("Val runs:", val_runs)
    print("Test runs:", test_runs)

    train_df = df[df["run"].isin(train_runs)].copy()
    val_df = df[df["run"].isin(val_runs)].copy()
    test_df = df[df["run"].isin(test_runs)].copy()

    return train_df, val_df, test_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arch", type=str, default="small_cnn", choices=["small_cnn", "resnet18"])
    ap.add_argument("--data_path", type=str, default=None, help="Path to CSV with 'image_path' and 'label' columns")
    ap.add_argument("--root_dir", type=str, default="data", help="Root directory for image paths in the CSV")
    ap.add_argument("--img_size", type=int, default=224, help="Image size to resize to")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    ap.add_argument("--use_class_weights", action="store_true", help="Compute class weights for CrossEntropyLoss from training labels")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for grouped split")
    args = ap.parse_args()
    set_seed(args.seed)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_weights = None

    # Create synthetic dataset for demo if no real data provided
    if args.data_path is None:
        print("No data path provided. Creating synthetic dataset for demo...")
        os.makedirs("checkpoints", exist_ok=True)

        X_train = torch.randn(160, 1, args.img_size, args.img_size)
        y_train = torch.randint(0, 2, (160,))
        X_val = torch.randn(40, 1, args.img_size, args.img_size)
        y_val = torch.randint(0, 2, (40,))

        ds_train = TensorDataset(X_train, y_train)
        ds_val = TensorDataset(X_val, y_val)

    else:
        print(f"Loading real data from {args.data_path}")
        import pandas as pd
        from data.impedance_dataset import CrackDataset

        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"CSV file not found: {args.data_path}")

        df = pd.read_csv(args.data_path)
        required_cols = {"image_path", "label"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {required_cols}. Found: {list(df.columns)}")

        train_df, val_df, test_df = grouped_split_by_run(
    df,
    args.train_ratio,
    args.val_ratio,
    1 - args.train_ratio - args.val_ratio,
    seed=args.seed
)

        # Check for missing image files
        missing = []
        for subset_df in (train_df, val_df, test_df):
            for p in subset_df['image_path'].astype(str).tolist():
                fp = os.path.join(args.root_dir, p)
                if not os.path.exists(fp):
                    missing.append(fp)
        if missing:
            sample_missing = missing[:5]
            raise FileNotFoundError(
                f"{len(missing)} image files referenced in CSV were not found. "
                f"Examples: {sample_missing}"
            )

        t_train = build_transforms(args.img_size, True, None, True)
        t_val = build_transforms(args.img_size, True, None, False)

        ds_train = CrackDataset(train_df, args.root_dir, transform=t_train)
        ds_val = CrackDataset(val_df, args.root_dir, transform=t_val)
        ds_test = CrackDataset(test_df, args.root_dir, transform=t_val)

        if args.use_class_weights:
            import numpy as _np
            labels = train_df['label'].astype(int).values
            classes, counts = _np.unique(labels, return_counts=True)
            total = labels.shape[0]
            weights = [total / (len(classes) * c) for c in counts]
            class_weights = torch.tensor(weights, dtype=torch.float)
            print(f"Using class weights: {class_weights}")
        else:
            class_weights = None

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train samples: {len(ds_train)}, Val samples: {len(ds_val)}")

    model = build_model(args.arch, num_classes=2, pretrained=False, dropout=0.2, grayscale=True).to(device)
    print(f"Model: {type(model).__name__}")

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    # initialize history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, dl_train, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, dl_val, criterion, device, epoch)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_acc": val_acc}, "checkpoints/best.pt")
            print(f"  ✅ New best model saved (val_acc={val_acc:.4f})")

    # save training history
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)
    print("Saved training history to training_history.json")

    # save test dataset
    if args.data_path is not None:
        torch.save(ds_test, "test_dataset.pt")
        print("Saved test dataset to test_dataset.pt")

    # generate predictions + labels
    if args.data_path is not None:
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for imgs, labels in dl_test:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        torch.save({"preds": all_preds, "labels": all_labels}, "test_results.pt")
        print("Saved test predictions + labels to test_results.pt")

    print("\n" + "="*60)
    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")
    print("Checkpoint saved to: checkpoints/best.pt")
    print("="*60)


if __name__ == "__main__":
    main() 
    