# test_model.py

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

# Add project root to sys.path so modules can be imported
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

from data.impedance_dataset import CrackDataset, load_splits, build_transforms
from models.crack_classifier import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    # Demo mode: create a dummy model and test it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = build_model("small_cnn", num_classes=2, pretrained=False, dropout=0.2, grayscale=True).to(device)
    print(f"Model built: {type(model).__name__}")

    # Create a dummy batch and test forward pass
    dummy_input = torch.randn(4, 1, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Model forward pass successful!")

if __name__ == "__main__":
    main()