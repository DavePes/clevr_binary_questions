import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise
from keras.layers import Input, Dense, Concatenate,Dropout
from keras.models import Model
import keras
import numpy as np
import torch
from questionLoading import BinaryQuestionHandler as ql
def load_features(location):
    data = np.load(f"features/{location}/consolidated_features.npz")
    return data["image_features"], data["text_features"], data["labels"]

train_image_features, train_text_features, train_labels = load_features("train")
val_image_features, val_text_features, val_labels = load_features("val")

# Define the classifier model
def build_classifier(input_dim,dropout_rate=0.4):
    # Inputs for image and text features
    image_input = Input(shape=(input_dim,), name="image_input")
    text_input = Input(shape=(input_dim,), name="text_input")
    
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

# Build the classifier
input_dim = train_image_features.shape[1]  # both image and text features have the same dimension
classifier = build_classifier(input_dim)

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