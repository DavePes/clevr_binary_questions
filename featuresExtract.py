from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()  # Set CLIP to evaluation mode

def extract_features(location):
    number = 
    while True:

        image_path = f"CLEVR_v1.0\images\{location}\CLEVR_{location}_{number}.png"
        break
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    images = [Image.open(img).convert("RGB") for img in image_paths]

    inputs = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Check the shape
    print("Shape of batch image features:", image_features.shape)

# Feature extraction for a batch of image-text pairs
def extract_features(location, save_dir="features/{location}"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (image_path, question) in enumerate(zip(image_paths, questions)):
        # Preprocess image and text
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=[question], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds.cpu().numpy()
            text_features = outputs.text_embeds.cpu().numpy()

        # Save features
        np.save(os.path.join(save_dir, f"image_{idx}.npy"), image_features)
        np.save(os.path.join(save_dir, f"text_{idx}.npy"), text_features)

# Example usage
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]  # Replace with image paths
questions = ["Is there a red sphere?", "Is the cube blue?", "Is there a yellow cylinder?"]
extract_features(image_paths, questions)
