import os
from PIL import Image

# Folder where your original graphs are
input_dir = r"C:\Users\Porte\OneDrive\Documents\Impedance Graphs For CNN"

# Folder where cropped images will be saved
output_dir = r"C:\Users\Porte\OneDrive\Documents\Impedance Graphs For CNN\PNG_Cropped"

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Crop settings
CROP_HEIGHT = 421          # final height
TOP_CROP = 480 - CROP_HEIGHT   # remove 59 pixels from the TOP

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(input_dir, filename)
        img = Image.open(path)

        # Crop from the TOP
        cropped = img.crop((0, TOP_CROP, img.width, img.height))

        save_path = os.path.join(output_dir, filename)
        cropped.save(save_path)

print("Cropping complete.")