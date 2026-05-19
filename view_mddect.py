import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load(r"C:\Users\Porte\OneDrive\Documents\MDDECTtest.npy")

# Inspect shape and dtype
print("Data shape:", data.shape)   # (3, 8, 2, 5, 20, 1250, 2)
print("Data type:", data.dtype)

# Map original classes (0,1,2) to binary labels (change from 3 labels to 2, 0 is no crack, 1 and 2 are crack) 
# 0 stays 0, 1 and 2 become 1(binary classification)
binary_labels = {0: 0, 1: 1, 2: 1}

# Plot one sample from each original class, but show binary label
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for cls in range(data.shape[0]):  # loop through 3 original classes
    # Pick first angle, freq, run, sample
    sample = data[cls, 0, 0, 0, 0, :, :]
    real = sample[:, 0]
    imag = sample[:, 1]

    # Get binary label
    label = binary_labels[cls]

    axes[cls].plot(real, imag)
    axes[cls].set_title(f"Original class {cls} → Binary label {label}")
    axes[cls].set_xlabel("Resistance")
    axes[cls].set_ylabel("Reactance")

plt.tight_layout()
plt.show()