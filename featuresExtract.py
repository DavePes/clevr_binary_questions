from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np
import pickle
def load_binary_q(location):
    with open(f'binary_questions_{location}', 'rb') as f:
        binary_questions = pickle.load(f)
    return binary_questions



def extract_features(location,batch_size=200):
    save_dir = f"features/{location}"
    full_questions = load_binary_q(location)
    questions = [full_question[1][0] for full_question in full_questions]
    question_indices = [full_question[0] for full_question in full_questions]
    image_dir = f"CLEVR_v1.0/images/{location}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])  # Sort ensures ascending order
    image_paths = [os.path.join(image_dir, img) for img in image_files]
    valid_image_paths = [image_paths[idx] for idx in question_indices]

    total_images = len(valid_image_paths)
    print(f"Total images: {total_images}, Processing in batches of {batch_size}")

    for start_idx in range(0, len(valid_image_paths), batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        batch_images = [Image.open(valid_image_paths[i]).convert("RGB") for i in range(start_idx, end_idx)]
        batch_questions = questions[start_idx:end_idx]

        # Preprocess and pass through the model
        inputs = processor(images=batch_images, text=batch_questions, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds.cpu().numpy()  # Shape: (batch_size, 512)
            text_features = outputs.text_embeds.cpu().numpy()    # Shape: (batch_size, 512)

        # Save features for the current batch
        for idx, i in enumerate(range(start_idx, end_idx)):
            np.save(os.path.join(save_dir, f"image_{i}.npy"), image_features[idx])
            np.save(os.path.join(save_dir, f"text_{i}.npy"), text_features[idx])

        print(f"Processed batch {start_idx}-{end_idx - 1}")

    print(f"Features saved in {save_dir}")
   

def consolidate_features(location):
    feature_dir = f"features/{location}"
    binary_q = ql.load_binary_q(location)
    answers = [q[1][1] for q in binary_q]
    labels = np.array(answers) == 'yes'
    num_samples = len(labels)

    # Assuming all features have the same shape
    image_shape = np.load(os.path.join(feature_dir, "image_0.npy")).shape
    text_shape = np.load(os.path.join(feature_dir, "text_0.npy")).shape

    image_features = np.zeros((num_samples, *image_shape), dtype=np.float32)
    text_features = np.zeros((num_samples, *text_shape), dtype=np.float32)

    for i in range(num_samples):
        image_path = os.path.join(feature_dir, f"image_{i}.npy")
        text_path = os.path.join(feature_dir, f"text_{i}.npy")
        if os.path.exists(image_path) and os.path.exists(text_path):
            image_features[i] = np.load(image_path)
            text_features[i] = np.load(text_path)

    np.savez_compressed(
        os.path.join(feature_dir, "consolidated_features.npz"),
        image_features=image_features,
        text_features=text_features,
        labels=labels
    )
if __name__ == '__main__':
    # # Ensure all imports and initializations happen inside the main block
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # model.eval()  # Set CLIP to evaluation mode

    # # Run feature extraction for the validation set in batches of 200
    # extract_features("train", batch_size=150)
    # extract_features('val', batch_size=150)
    # Call this once to consolidate features
    consolidate_features("train")
    consolidate_features("val")
