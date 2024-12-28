import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise
from keras.layers import Input, Dense, Concatenate,Dropout
from keras.models import Model
import keras
import numpy as np
import torch
from questionLoading import BinaryQuestionHandler as ql
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def load_features(location):
    data = np.load(f"features/{location}/consolidated_features.npz")
    return data["image_features"], data["text_features"], data["labels"]

# Load pre-trained DistilBERT
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')
model.eval()
def process_text_embeddings(input_texts, output_file, batch_size=200):
    # Check if the embeddings file already exists
    if os.path.exists(output_file):
        print(f"Loading embeddings from {output_file}")
        data = np.load(output_file)
        return data['text_features']  # Return the saved embeddings
    
    # If the embeddings file does not exist, generate them in batches
    print(f"Generating text embeddings and saving to {output_file}")
    
    num_samples = len(input_texts)
    all_embeddings = []  # List to store embeddings
    
    # Process in batches to avoid memory overload
    for start_idx in tqdm(range(0, num_samples, batch_size)):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_texts = input_texts[start_idx:end_idx]
        
        # Tokenize the batch of texts
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Run the batch through the DistilBERT model
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Take mean of token embeddings
            
        # Append batch embeddings to the list
        all_embeddings.append(embeddings)
    
    # Concatenate all embeddings into a single numpy array
    all_embeddings = np.vstack(all_embeddings)
    
    # Save the embeddings as a .npz file
    np.savez(output_file, text_features=all_embeddings)
    
    print(f"Embeddings saved to {output_file}")
    return all_embeddings

# Define the classifier model
def build_classifier(image_dim,text_dim,dropout_rate=0.25):
    # Inputs for image and text features
    image_input = Input(shape=(image_dim,), name="image_input")
    text_input = Input(shape=(text_dim,), name="text_input")
    
    # Combine features using Concatenate
    concatenated = Concatenate()([image_input, text_input])
    
    x = Dense(400)(concatenated) 
    x = keras.layers.BatchNormalization()(x) 
    x = keras.activations.relu(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(400)(concatenated) 
    x = keras.layers.BatchNormalization()(x) 
    x = keras.activations.relu(x) 
    x = Dropout(dropout_rate)(x)
    x = Dense(400)(concatenated) 
    x = keras.layers.BatchNormalization()(x) 
    x = keras.activations.relu(x) 
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation="sigmoid", name="output")(x)  # Binary classification output
    
    # Define the model
    model = Model(inputs=[image_input, text_input], outputs=output)
    return model

train_image_features, _, train_labels = load_features("train")
val_image_features, _, val_labels = load_features("val")

full_questions = ql.load_binary_q("train")  # Replace with your question data loading
questions = [full_question[1][0] for full_question in full_questions]  # Extract the question text
output_file = f"features/train/text_embeddings.npz"
train_text_features = process_text_embeddings(questions, output_file)

# Repeat for validation set
full_questions_val = ql.load_binary_q("val")
questions_val = [full_question[1][0] for full_question in full_questions_val]  # Extract the question text
output_file_val = f"features/val/text_embeddings.npz"
val_text_features = process_text_embeddings(questions_val, output_file_val)
# Build the classifier
image_dim = train_image_features.shape[1]
text_dim = train_text_features.shape[1]
classifier = build_classifier(image_dim,text_dim)

# Compile the model
classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)



# Model summary
classifier.summary()

# Train the classifier
history = classifier.fit(
    [train_image_features, train_text_features],  # Input: image and text features
    train_labels,  # Labels
    validation_data=([val_image_features, val_text_features], val_labels),
    batch_size=64,
    epochs=100
)

# Save the trained model
classifier.save("clip_classifier_model.keras")




def calculate_total_size(location):
    feature_dir = f"features/{location}"
    total_size = 0
    for dirpath, _, filenames in os.walk(feature_dir):
        for file in filenames:
            if file.endswith('.npy'):
                total_size += os.path.getsize(os.path.join(dirpath, file))
    return total_size / (1024 ** 3)  # Convert bytes to GB

# print("Train size:", calculate_total_size("train"))
# print("Val size:", calculate_total_size("val"))