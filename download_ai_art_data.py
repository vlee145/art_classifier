import os
from datasets import load_dataset, concatenate_datasets
from PIL import Image

# Create a directory to store the images
output_dir = "data/ai_art"
os.makedirs(output_dir, exist_ok=True)

# Load the datasets
dataset1 = load_dataset("sj21867/ai_art_exp1")
dataset2 = load_dataset("sj21867/ai_art_exp2")
dataset3 = load_dataset("sj21867/ai_art_exp3")

# Combine the datasets
combined_dataset = concatenate_datasets([dataset1['train'], dataset2['train'], dataset3['train']])

# Save the first 5 images
for i in range(10000):
    try:
        item = combined_dataset[i]
        if isinstance(item, dict) and 'image' in item:
            image = item['image']
            filename = os.path.join(output_dir, f"ai_art_{i+1}.png")
            image.save(filename)
            print(f"Saved: {filename}")
        else:
            print(f"Item {i+1} is not in the expected format")
    except Exception as e:
        print(f"Error processing item {i+1}: {str(e)}")

print(f"Images saved in the '{output_dir}' directory.")
