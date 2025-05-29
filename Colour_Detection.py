import cv2
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# Folder where raw images are stored
INPUT_FOLDER = "dataset_raw"
OUTPUT_FOLDER = "dataset_formatted"

# Define standard size
IMG_SIZE = (224, 224)

# Full car color dictionary (RGB values)
CAR_COLORS = {
    "Red": (255, 0, 0),
    "Dark Red": (139, 0, 0),
    "Maroon": (128, 0, 0),
    "Orange": (255, 165, 0),
    "Dark Orange": (255, 140, 0),
    "Yellow": (255, 255, 0),
    "Gold": (255, 215, 0),
    "Lime Green": (50, 205, 50),
    "Green": (0, 255, 0),
    "Dark Green": (0, 100, 0),
    "Cyan": (0, 255, 255),
    "Teal": (0, 128, 128),
    "Light Blue": (173, 216, 230),
    "Blue": (0, 0, 255),
    "Dark Blue": (0, 0, 139),
    "Navy": (0, 0, 128),
    "Purple": (128, 0, 128),
    "Magenta": (255, 0, 255),
    "Pink": (255, 192, 203),
    "White": (255, 255, 255),
    "Silver": (192, 192, 192),
    "Gray": (128, 128, 128),
    "Dark Gray": (64, 64, 64),
    "Black": (0, 0, 0),
    "Brown": (165, 42, 42),
    "Beige": (245, 245, 220)
}

# Create output folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to detect dominant color
def get_dominant_color(image, k=3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image.reshape((-1, 3))  # Flatten image
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image)
    dominant_color = kmeans.cluster_centers_[0]  # Most dominant cluster
    return dominant_color

# Function to find the closest named color
def get_closest_color(rgb):
    return min(CAR_COLORS.keys(), key=lambda c: np.linalg.norm(rgb - np.array(CAR_COLORS[c])))

# Prepare metadata list
metadata = []

# Process each image
for filename in tqdm(os.listdir(INPUT_FOLDER)):
    img_path = os.path.join(INPUT_FOLDER, filename)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Unreadable image: {filename}")  # Print unreadable image name
        continue  # Skip processing for this file

    # Resize image
    img_resized = cv2.resize(img, IMG_SIZE)

    # Convert to HSV and extract dominant color
    dominant_color = get_dominant_color(img_resized)
    
    # Find closest named color
    closest_color = get_closest_color(dominant_color)

    # Save processed image
    new_filename = filename.split(".")[0] + ".jpg"
    new_path = os.path.join(OUTPUT_FOLDER, new_filename)
    cv2.imwrite(new_path, img_resized)

    # Extract car details from filename (if structured like "toyota_camry_2018.jpg")
    parts = filename.lower().replace(".jpg", "").replace("_", " ").split()
    if len(parts) >= 3:
        make, model, year = parts[0], parts[1], parts[2]
    else:
        make, model, year = "Unknown", "Unknown", "Unknown"

    # Store metadata
    metadata.append([new_path, make, model, year, closest_color])

# Save metadata to CSV
df = pd.DataFrame(metadata, columns=["image_path", "make", "model", "year", "color"])
df.to_csv("dataset_metadata.csv", index=False)

print("Dataset formatted successfully!")
