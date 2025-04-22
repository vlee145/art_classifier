import deeplake
import os
from PIL import Image
import numpy as np

# Load the dataset from Deep Lake
ds = deeplake.load('hub://activeloop/wiki-art')

# Create a directory named 'human_art' to save images
os.makedirs("data/human_art", exist_ok=True)

# Download the first 10 images
for i, sample in enumerate(ds[:10100]):
    # Access the image array
    image_array = sample['images'].numpy()  # Convert to numpy array

    # Use the 'labels' tensor for the label
    label = sample['labels'].data()  # Get the label directly

    # Convert the numpy array to an image
    image = Image.fromarray(image_array)

    # Save the image in the 'human_art' folder
    image.save(f"data/human_art_evalutaion/image_{i+1}_{label}.jpg")
    print(f"Saved data/human_art/image_{i+1}_{label}.jpg")

