from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np
import pickle
from questionLoading import BinaryQuestionHandler as ql
import matplotlib.pyplot as plt

def load_binary_q(location):
    with open(f'binary_questions_{location}', 'rb') as f:
        binary_questions = pickle.load(f)
    return binary_questions
def show_images_and_questions(location="train"):
    save_dir = f"raw/{location}"
    path = f"{save_dir}/data.npz"  # Path to save the combined .npz
    # Load the images and questions from the .npz files
    data = np.load(path, allow_pickle=True)
    images = data['images']
    questions = data["questions"]
    
    total_images = len(images)
    print(f"Displaying {total_images} images with questions.")
    
    # Iterate through all the images and questions
    for i in range(total_images):
        image = images[i]
        question = questions[i]
        
        # Convert the image from numpy array back to an image
        image = Image.fromarray(image)
        
        # Display the image and the question
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')  # Hide axis
        plt.title(f"Question: {question}", fontsize=14)
        plt.show()
        
        # Optionally, wait for user input to move to the next image
        #input("Press Enter to show next image...")


def extract_features(location,batch_size=500):
    save_dir = f"features/{location}"
    if (os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0):
        print("ending")
        return
    full_questions = load_binary_q(location)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()  # Set CLIP to evaluation mode
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

def features_without_change(location, batch_size=500):
    save_dir = f"raw/{location}"
    save_path = f"{save_dir}/data.npz"  # Path to save the combined .npz
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    
    full_questions = load_binary_q(location)  # Load questions
    questions = [full_question[1] for full_question in full_questions]  # Extract all questions
    question_indices = [full_question[0] for full_question in full_questions]  # Extract image indices
    image_dir = f"CLEVR_v1.0/images/{location}"
    
    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    image_paths = [os.path.join(image_dir, img) for img in image_files]
    
    # Select valid image paths based on question indices
    valid_image_paths = [image_paths[idx] for idx in question_indices]
    total_images = len(valid_image_paths)
    print(f"Total images: {total_images}, Processing in batches of {batch_size}")
    
    # Step 1: Save Questions
    np.savez_compressed(save_path, questions=np.array(questions, dtype=object))
    print(f"Questions saved in {save_path}")
    
    # Step 2: Preallocate Image Array
    # Assuming images are of fixed size (320, 480, 3)
    image_shape = (320, 480, 3)  # Update if images have a different fixed size
    accumulated_images = np.empty((total_images, *image_shape), dtype=np.uint8)
    
    # Step 3: Fill Preallocated Array
    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        for i, idx in enumerate(range(start_idx, end_idx)):
            image_path = valid_image_paths[idx]
            img = Image.open(image_path).convert("RGB")
            accumulated_images[start_idx + i] = np.array(img, dtype=np.uint8)  # Fill the preallocated array
        
        print(f"Processed batch {start_idx}-{end_idx - 1}")
    
    # Save Images
    with np.load(save_path, allow_pickle=True) as data:
        questions = data["questions"]
    print("compressed saving")
    np.savez_compressed(save_path, questions=questions, images=accumulated_images)
    print(f"Images saved in {save_path}")



if __name__ == '__main__':

    # # Run feature extraction for the validation set in batches of 200

    extract_features("train", batch_size=500)
    extract_features('val', batch_size=500)
    # #Call this once to consolidate features
    consolidate_features("train")
    consolidate_features("val")
    # Example call to the function
    features_without_change("train", batch_size=2000)
    #show_images_and_questions("train")
    features_without_change("val", batch_size=500)
