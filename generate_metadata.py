import os
import csv

image_dir = r"C:\Users\Porte\OneDrive\Documents\Impedance Graphs For CNN\PNG_Cropped"
output_csv = r"C:\Users\Porte\OneDrive\Documents\Impedance Graphs For CNN\metadata.csv"

def extract_run(filename):
    return int(filename.split("_run")[1].split("_")[0])

rows = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith(".png"):
        parts = filename.split("_")
        class_part = parts[0]
        class_num = int(class_part.replace("class", ""))

        label = 0 if class_num == 0 else 1
        run = extract_run(filename)

        rows.append([filename, label, run])

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label", "run"])
    writer.writerows(rows)

print("metadata.csv created successfully.")