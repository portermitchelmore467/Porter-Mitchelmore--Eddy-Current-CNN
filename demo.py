import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button

import numpy as np
import cv2

# Config 
MODEL_PATH = r"C:\PorterCapstoneCNN\scripts\checkpoints\best.pt"
print("Loading checkpoint from:", MODEL_PATH)
IMG_SIZE = 224


# Load model architecture
from models.crack_classifier import SmallCNN

device = torch.device("cpu")

model = SmallCNN(num_classes=2, dropout=0.2, in_channels=1)
checkpoint = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()


# Preprocessing (to match training)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# GUI Setup
root = tk.Tk()
root.title("Crack Detection CNN Demo")
root.geometry("650x650")

title = Label(root, text="Microcrack Detection System", font=("Arial", 20, "bold"))
title.pack(pady=10)

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

confidence_label = Label(root, text="", font=("Arial", 14))
confidence_label.pack()

def autocrop_loop_only(pil_img):
    """
    Enhances the loop and isolates it using contrast boosting + edge detection.
    Removes axes, labels, and gridlines much more effectively.
    """
    # Convert PIL → OpenCV grayscale
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- Step 1: Boost contrast (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # --- Step 2: Sharpen the loop edges ---
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)

    # --- Step 3: Edge detection ---
    edges = cv2.Canny(sharp, threshold1=20, threshold2=60)

    # Find non-zero edge pixels
    ys, xs = np.where(edges > 0)

    if len(xs) == 0 or len(ys) == 0:
        return pil_img  # fallback

    # Bounding box around the loop
    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()

    # Add padding
    pad = 10
    left = max(0, left - pad)
    right = min(img.shape[1], right + pad)
    top = max(0, top - pad)
    bottom = min(img.shape[0], bottom + pad)

    # Crop and return as PIL
    cropped = img[top:bottom, left:right]
    return Image.fromarray(cropped)



# Prediction Function
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )

    if not file_path:
        return

    # Load the image (FULL IMAGE, NO CROPPING) important!!!!
    img = Image.open(file_path).convert("RGB")


    # Display the full image
    display_img = img.resize((420, 280))
    img_tk = ImageTk.PhotoImage(display_img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    # Preprocess (matches training)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    confidence = confidence.item()
    prediction = prediction.item()

    # Display result
    if prediction == 1:
        result_label.config(text="CRACK DETECTED", fg="red")
    else:
        result_label.config(text="NO CRACK DETECTED", fg="green")

    confidence_label.config(text=f"Confidence: {confidence * 100:.2f}%")


# Button
upload_button = Button(
    root,
    text="Upload Image",
    command=upload_and_predict,
    font=("Arial", 14),
    width=18
)
upload_button.pack(pady=20)


separator = Label(root, text="―" * 40, fg="lightgray")
separator.pack(side="bottom", pady=(0,2))

footer = Label(root, text="Demo created by Porter Mitchelmore (2026)", font=("Arial", 10, "italic"), fg="red")
footer.pack(side="bottom", pady=5)
root.mainloop()
