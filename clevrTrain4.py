import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from timm import create_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Custom Dataset for PyTorch
class FeatureDataset(Dataset):
    def __init__(self, images, text_inputs, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.text_inputs = text_inputs  # Expecting a dictionary with 'input_ids' and 'attention_mask'
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.images[idx].permute(2, 0, 1),  # Convert (H, W, C) -> (C, H, W)
                self.text_inputs['input_ids'][idx], 
                self.text_inputs['attention_mask'][idx], 
                self.labels[idx])

# Function to load features
def load_features(location):
    labels = np.load(f"features/{location}/consolidated_features.npz")["labels"]
    data = np.load(f"raw/{location}/data.npz", allow_pickle=True,mmap_mode='r')
    questions = data['questions'][:, 0]
    images = data['images']
    return images, questions, labels

# Define the classifier model
class Classifier(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim=400, dropout_rate=0.25):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(image_dim + text_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, image_features, text_features):
        x = torch.cat((image_features, text_features), dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return torch.sigmoid(x)

def compute_accuracy(predictions, labels):
    # Threshold the sigmoid outputs at 0.5 to get binary predictions
    predicted_labels = (predictions >= 0.5).float()
    correct = (predicted_labels == labels).float()
    accuracy = correct.sum() / correct.size(0)
    return accuracy

def main():
    # Load DistilBERT
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)


    # Load MobileViT-Small pretrained model
    mobilevit_model = create_model('mobilevit_s', pretrained=True)
    mobilevit_model.classifier = nn.Identity()  # Remove the classification head to get feature embeddings
    mobilevit_model = mobilevit_model.to(device)
    
    # Load raw data
    train_images, train_questions, train_labels = load_features("train")
    val_images, val_questions, val_labels = load_features("val")

    # Tokenize text questions
    train_text_inputs = tokenizer(list(train_questions), padding=True, truncation=True, return_tensors="pt").to(device)
    val_text_inputs = tokenizer(list(val_questions), padding=True, truncation=True, return_tensors="pt").to(device)

    # Create datasets and dataloaders
    train_dataset = FeatureDataset(train_images, train_text_inputs, train_labels)
    val_dataset = FeatureDataset(val_images, val_text_inputs, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model setup
    image_dim = mobilevit_model.embed_dim
    text_dim = bert_model.config.hidden_size
    classifier = Classifier(image_dim, text_dim).to(device)

    # Separate optimizers with different learning rates
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    bert_optimizer = optim.Adam(bert_model.parameters(), lr=2e-5)
    mobilevit_optimizer = optim.Adam(mobilevit_model.parameters(), lr=1e-5)

    criterion = nn.BCELoss()
    #scaler = torch.cuda.amp.GradScaler()
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        classifier.train()
        bert_model.train()
        total_loss = 0
        total_accuracy = 0  # Variable to accumulate accuracy

        for images, input_ids, attention_mask, labels in tqdm(train_loader):
            images, input_ids, attention_mask, labels = (
                images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            )
            # Forward pass through MobileViT
            image_features = mobilevit_model(images)

            # Forward pass through DistilBERT
            text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            text_features = text_outputs[:, 0, :]  # Use [CLS] token representation

            # Forward pass through classifier
            classifier_optimizer.zero_grad()
            bert_optimizer.zero_grad()
            mobilevit_optimizer.zero_grad()

            outputs = classifier(image_features, text_features)
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            classifier_optimizer.step()
            bert_optimizer.step()
            mobilevit_optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(classifier_optimizer)
            # scaler.step(bert_optimizer)
            # scaler.update()
            total_loss += loss.item()

            # Calculate accuracy
            accuracy = compute_accuracy(outputs.squeeze(), labels)
            total_accuracy += accuracy.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {total_accuracy/len(train_loader):.4f}")

        # Validation loop
        classifier.eval()
        bert_model.eval()
        mobilevit_model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, input_ids, attention_mask, labels in val_loader:
                images, input_ids, attention_mask, labels = images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
                
                # Extract image features
                image_features = mobilevit_model(images)

                # Extract text features
                text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                text_features = text_outputs[:, 0, :]

                # Forward pass through classifier
                outputs = classifier(image_features, text_features)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                # Calculate accuracy
                accuracy = compute_accuracy(outputs.squeeze(), labels)
                val_accuracy += accuracy.item()

            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy/len(val_loader):.4f}")
        torch.cuda.empty_cache()
    # Save the trained model
    torch.save(classifier.state_dict(), "clip_classifier_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    main()
