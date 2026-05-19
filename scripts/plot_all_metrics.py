import json
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_curve, auc
import numpy as np
import os
import sys
sys.path.append("..")

# ---------------------------------------------------------
# Load training history
# ---------------------------------------------------------
with open("training_history.json", "r") as f:
    history = json.load(f)

train_loss = history["train_loss"]
val_loss = history["val_loss"]
train_acc = history["train_acc"]
val_acc = history["val_acc"]

# ---------------------------------------------------------
# Plot Accuracy Curve
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot Loss Curve
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Load test predictions + labels
# ---------------------------------------------------------
test_results = torch.load("test_results.pt", weights_only=False)
preds = np.array(test_results["preds"])
labels = np.array(test_results["labels"])

# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(6,6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Classification Report

report = classification_report(labels, preds)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)


# ROC Curve 
# ---------------------------------------------------------
# Convert labels to 0/1
labels_bin = labels

# Load model to get probabilities
checkpoint = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model_state = checkpoint["model"]

# You must import your model architecture
from models.crack_classifier import build_model
model = build_model("small_cnn", num_classes=2, pretrained=False, dropout=0.2, grayscale=True)
model.load_state_dict(model_state)
model.eval()

# Load test dataset
ds_test = torch.load("test_dataset.pt", weights_only=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False)

probs = []
with torch.no_grad():
    for imgs, _ in dl_test:
        logits = model(imgs)
        softmax = torch.softmax(logits, dim=1)
        probs.extend(softmax[:,1].numpy())  # probability of class 1

probs = np.array(probs)

fpr, tpr, _ = roc_curve(labels_bin, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()

print("All graphs generated successfully.")