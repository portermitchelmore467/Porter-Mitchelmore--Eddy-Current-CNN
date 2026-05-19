import numpy as np
import matplotlib.pyplot as plt
import os

# Load dataset
data = np.load(r"C:\Users\Porte\OneDrive\Documents\MDDECTtest.npy")

# Create output folder (will be created if it doesn't exist)
output_dir = r"C:\Users\Porte\OneDrive\Documents\Impedance Graphs For CNN"
os.makedirs(output_dir, exist_ok=True)

# Loop through all dimensions
for cls in range(data.shape[0]):       # 2 classes
    for angle in range(data.shape[1]): # 8 angles
        for freq in range(data.shape[2]):  # 2 frequencies
            for run in range(data.shape[3]):   # 5 runs
                for sample in range(data.shape[4]): # 20 samples
                    curve = data[cls, angle, freq, run, sample, :, :]
                    real = curve[:, 0]
                    imag = curve[:, 1]

                    plt.figure()
                    plt.plot(real, imag)
                    plt.xlabel("Resistance")
                    plt.ylabel("Reactance")
                    plt.title(f"Class {cls}, Angle {angle}, Freq {freq}, Run {run}, Sample {sample}")

                    # Save each graph with descriptive filename
                    filename = f"class{cls}_angle{angle}_freq{freq}_run{run}_sample{sample}.png"
                    plt.savefig(os.path.join(output_dir, filename))
                    plt.close()

print("✅ All graphs saved to:", output_dir)