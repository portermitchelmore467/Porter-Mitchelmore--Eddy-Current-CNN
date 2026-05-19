# inspect_import.py - Quick inspection of available modules

import sys
from pathlib import Path

# Add project root to path
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

print("=" * 60)
print("IMPORT INSPECTION")
print("=" * 60)

# Test core imports
try:
    from data.impedance_dataset import CrackDataset, load_splits, build_transforms
    print("✅ data.impedance_dataset imported successfully")
except Exception as e:
    print(f"❌ data.impedance_dataset import failed: {e}")

try:
    from models.crack_classifier import build_model, SmallCNN
    print("✅ models.crack_classifier imported successfully")
except Exception as e:
    print(f"❌ models.crack_classifier import failed: {e}")

# Test model instantiation
try:
    import torch
    model = build_model("small_cnn", num_classes=2, pretrained=False, grayscale=True)
    dummy = torch.randn(2, 1, 224, 224)
    out = model(dummy)
    print(f"✅ SmallCNN forward pass: input {dummy.shape} -> output {out.shape}")
except Exception as e:
    print(f"❌ SmallCNN forward pass failed: {e}")

# Test ResNet18 import
try:
    model = build_model("resnet18", num_classes=2, pretrained=False, grayscale=True)
    dummy = torch.randn(2, 1, 224, 224)
    out = model(dummy)
    print(f"✅ ResNet18 forward pass: input {dummy.shape} -> output {out.shape}")
except Exception as e:
    print(f"❌ ResNet18 forward pass failed: {e}")

print("=" * 60)
