import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data.impedance_dataset import CrackDataset, build_transforms
from models.crack_classifier import build_model
from train_model import grouped_split_by_run

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = r"C:\PorterCapstoneCNN\scripts\metadata.csv"
ROOT_DIR = r"C:\Users\Porte\OneDrive\Documents\Impedance_Graphs_For_CNN\PNG_Cropped"
MODEL_PATH = r"C:\PorterCapstoneCNN\scripts\checkpoints\best.pt"
IMG_SIZE = 224

# -----------------------------
# Load CSV + Test Split
# -----------------------------
df = pd.read_csv(CSV_PATH)
_, _, test_df = grouped_split_by_run(df, seed=42)

print("Test samples:", len(test_df))

# -----------------------------
# Dataset + Loader
# -----------------------------
transform = build_transforms(IMG_SIZE, grayscale=True, augment_cfg=None, is_train=False)
test_ds = CrackDataset(test_df, ROOT_DIR, transform=transform)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cpu")
model = build_model("small_cnn", num_classes=2, pretrained=False, dropout=0.2, grayscale=True)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# -----------------------------
# Inference
# -----------------------------
all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probability of "crack" and rate of correctness 
        preds = torch.argmax(logits, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Crack", "Crack"],
            yticklabels=["No Crack", "Crack"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["No Crack", "Crack"]))

# -----------------------------
# ROC Curve + AUC
# -----------------------------
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print(f"\nAUC: {roc_auc:.4f}")