import os
import pandas as pd

# Paths
csv_path = r"C:\Users\Porte\OneDrive\Documents\Impedance Graphs For CNN\metadata.csv"
root_dir = r"C:\Users\Porte\OneDrive\Documents\Impedance Graphs For CNN\PNG_Cropped"

# Load CSV
df = pd.read_csv(csv_path)

# Build full paths
df["full_path"] = df["image_path"].apply(lambda x: os.path.join(root_dir, x))

# Check existence
df["exists"] = df["full_path"].apply(os.path.exists)

# Filter only existing files
clean_df = df[df["exists"]].copy()

# Drop helper columns
clean_df = clean_df.drop(columns=["full_path", "exists"])

# Save cleaned CSV
output_path = r"C:\Users\Porte\OneDrive\Documents\Impedance Graphs For CNN\metadata_cleaned.csv"
clean_df.to_csv(output_path, index=False)

print(f"Original rows: {len(df)}")
print(f"Rows kept: {len(clean_df)}")
print(f"Rows removed: {len(df) - len(clean_df)}")
print(f"Cleaned CSV saved to: {output_path}")