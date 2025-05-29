import os

# Path to your dataset folder
dataset_path = "C:/Users/Lenovo/Desktop/Car Detection AI/DATA/data_2/images.cv_33mdprbld2vsvfc8l3obl/data/train"

# Get list of brand names (folder names)
brands = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
num_brands = len(brands)

print(f"Number of brands: {num_brands}")
print("Brands:", brands)

# Save to a file for reference (optional)
with open("brands.txt", "w") as f:
    f.write("\n".join(brands))