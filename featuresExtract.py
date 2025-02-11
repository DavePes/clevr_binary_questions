from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np
import pickle
from questionLoading import BinaryQuestionHandler as ql
import matplotlib.pyplot as plt
import argparse
import random
import torchvision.transforms as T

def load_binary_q(location):
    if not os.path.exists(f'binary_questions_{location}'):
        ql.save_binary_q(location)
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


def extract_features(location, batch_size=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = f"features/{location}"
    full_questions = load_binary_q(location)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    questions = [q[1][0] for q in full_questions]
    question_indices = [q[0] for q in full_questions]
    image_dir = f"CLEVR_v1.0/images/{location}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    image_paths = [os.path.join(image_dir, img) for img in image_files]
    valid_image_paths = [image_paths[idx] for idx in question_indices]

    total_images = len(valid_image_paths)
    print(f"Total images: {total_images}, Processing in batches of {batch_size}")

    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        batch_images = [Image.open(valid_image_paths[i]).convert("RGB") for i in range(start_idx, end_idx)]
        batch_questions = questions[start_idx:end_idx]

        inputs = processor(images=batch_images, text=batch_questions, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds.cpu().numpy()
            text_features = outputs.text_embeds.cpu().numpy()

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
def question_label_save(full_questions,save_path):
    label_questions = []
    for one_image_questions in full_questions:
        one_image_questions_array = np.array(one_image_questions[1:])
        one_image_questions_array[:,1] = one_image_questions_array[:,1] == 'yes'
        label_questions.append(one_image_questions_array)
    np.savez_compressed(save_path,np.array(label_questions, dtype=object),allow_pickle=True)

def features_without_change(location, batch_size=500):
    save_dir = f"raw/{location}"
    save_path = f"{save_dir}"  # Path to save
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    full_questions = load_binary_q(location)  # Load questions
    question_indices = [full_question[0] for full_question in full_questions]  # Extract image indices

    question_label_save(full_questions,save_path+"/ql.npz")

    image_dir = f"CLEVR_v1.0/images/{location}"
    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    image_paths = [os.path.join(image_dir, img) for img in image_files]
    
    # Select valid image paths based on question indices
    valid_image_paths = [image_paths[idx] for idx in question_indices]
    total_images = len(valid_image_paths)
    print(f"Total images: {total_images}, Processing in batches of {batch_size}")
    
    
    # Preallocate Image Array
    # Assuming images are of fixed size (320, 480, 3)
    ## crop -> then resize to 0.6
    #new_size = (276, 180)
    ## NEW RESIZING INTO 224*224
    new_size = (224, 224)
    accumulated_images = np.empty((total_images, *new_size[::-1], 3), dtype=np.uint8)
    # bicubic interpolation
    # Fill Preallocated Array
    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        for i, idx in enumerate(range(start_idx, end_idx)):
            image_path = valid_image_paths[idx]
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            cropped_img = img.crop((10, 10, width - 10, height - 10))  # (left, top, right, bottom)
                
            # Resize the cropped image using bicubic interpolation
            width,height = cropped_img.size
            resized_img = cropped_img.resize(new_size, Image.BICUBIC)   
            #resized_img = img.resize(new_size, Image.BICUBIC)
            accumulated_images[start_idx + i] = np.array(resized_img, dtype=np.uint8)  # Fill the preallocated array
        
        print(f"Processed batch {start_idx}-{end_idx - 1}")
    print("compressed saving")
    np.savez_compressed(save_path + "/images.npz",images=accumulated_images)
    print(f"Images saved in {save_path}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run feature extraction and related tasks.")
    parser.add_argument("--function", type=str, required=True, choices=[
        "extr", "cons", "no_change", "show"],
        help="The function to execute.")
    parser.add_argument("--location", type=str, required=True, help="Dataset location (e.g., 'train' or 'val').")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for processing.")

    args = parser.parse_args()

    if args.function == "extr":
        extract_features(args.location, args.batch_size)
    elif args.function == "cons":
        consolidate_features(args.location)
    elif args.function == "no_change":
        features_without_change(args.location, args.batch_size)
    elif args.function == "show":
        show_images_and_questions(args.location)
        
# if __name__ == '__main__':

#     # # Run feature extraction for the validation set in batches of 200

#     extract_features("train", batch_size=500)
#     extract_features('val', batch_size=500)
#     # #Call this once to consolidate features
#     consolidate_features("train")
#     consolidate_features("val")
#     # Example call to the function
#     features_without_change("train", batch_size=2000)
#     #show_images_and_questions("train")
#     features_without_change("val", batch_size=500)
