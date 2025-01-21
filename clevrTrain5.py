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

class FeatureDataset(Dataset):
    def __init__(self, data_path, labels, tokenizer):
        """
        data_path: Path to the .npz file containing 'images' and 'questions'.
        labels: NumPy array of labels (float 0/1 or similar).
        tokenizer: A tokenizer from Transformers (e.g., AutoTokenizer).
        """
        # Memory-map the NPZ so we don't load everything at once
        self.data = np.load(data_path, mmap_mode='r',allow_pickle=True)
        
        # These are still NumPy arrays, but memory-mapped
        self.images = self.data['images']            # shape: (N, H, W, C), dtype=uint8
        self.questions = self.data['questions'][:, 0]  # shape: (N,), e.g. each entry is a string or object
        self.labels = labels                          # shape: (N,), separate np array
        self.tokenizer = tokenizer
        
        # Store the length
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1) Load image from disk (uint8)
        image_uint8 = self.images[idx]  # shape (H, W, C)
        
        # 2) Convert to float32 and normalize
        image_tensor = torch.tensor(image_uint8, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # image_tensor shape: (C, H, W)
        
        # 3) Retrieve question as string
        #question_str = str(self.questions[idx])
        question_str = self.questions[idx]
        # 4) Tokenize question
        #    Return_tensors='pt' -> shape: (1, seq_len)
        encoded = self.tokenizer(
            question_str,
            #padding="max_length",
            padding=True,
            truncation=True,
            #max_length=32,        # or whatever
            return_tensors="pt"
        )
        
        # Squeeze out the batch dimension (1)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # 5) Convert label to float tensor
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image_tensor, input_ids, attention_mask, label
  

# Define the classifier model
class Classifier(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim=400, dropout_rate=0.25):
        super(Classifier, self).__init__()
        print("Initializing Classifier...")
        self.fc1 = nn.Linear(image_dim + text_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        print("Classifier initialized.")

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
    predicted_labels = (predictions >= 0.5).float()
    correct = (predicted_labels == labels).float()
    accuracy = correct.sum() / correct.size(0)
    return accuracy

def main():
    # 1) Load the tokenizer (e.g., DistilBERT)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 2) Load models
    bert_model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    mobilevit_model = create_model('mobilevit_s', pretrained=True)
    mobilevit_model.classifier = nn.Identity()
    mobilevit_model = mobilevit_model.to(device)
  
    # 3) Load labels
    labels_train = np.load("features/train/consolidated_features.npz")["labels"]
    labels_val = np.load("features/val/consolidated_features.npz")["labels"]

    # 4) Create the Dataset
    train_dataset = FeatureDataset(
        data_path="raw/train/data.npz",
        labels=labels_train,
        tokenizer=tokenizer
    )
    val_dataset = FeatureDataset(
        data_path="raw/val/data.npz",
        labels=labels_val,          # <-- Use validation labels here
        tokenizer=tokenizer
    )

    # 5) Create the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,        #no need to shuffle
        num_workers=4,
        pin_memory=True
    )

    # 6) Prepare classifier
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Default input size for MobileViT
    dummy_output = mobilevit_model(dummy_input)
    image_dim = dummy_output.shape[1]
    print(f"Determined image_dim: {image_dim}")
    text_dim = bert_model.config.hidden_size
    classifier = Classifier(image_dim, text_dim).to(device)

    # 7) Define optimizers & loss
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    bert_optimizer = optim.Adam(bert_model.parameters(), lr=2e-5)
    mobilevit_optimizer = optim.Adam(mobilevit_model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()

    print("Setup complete.")

    # 8) Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        classifier.train()
        bert_model.train()
        mobilevit_model.train()

        total_loss = 0.0
        total_accuracy = 0.0

        for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            # Move inputs to device
            images, input_ids, attention_mask, labels = (
                images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            )

            # Zero out gradients
            classifier_optimizer.zero_grad()
            bert_optimizer.zero_grad()
            mobilevit_optimizer.zero_grad()

            # Forward pass
            image_features = mobilevit_model(images)  # shape: [batch_size, image_dim]
            text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            text_features = text_outputs[:, 0, :]     # [CLS] token

            outputs = classifier(image_features, text_features).squeeze()
            loss = criterion(outputs, labels)

            # Backprop
            loss.backward()
            classifier_optimizer.step()
            bert_optimizer.step()
            mobilevit_optimizer.step()

            # Accumulate stats
            total_loss += loss.item()
            accuracy = compute_accuracy(outputs, labels)
            total_accuracy += accuracy.item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_accuracy / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        # Validation
        print("Validating...")
        classifier.eval()
        bert_model.eval()
        mobilevit_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(val_loader):
                images, input_ids, attention_mask, labels = (
                    images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
                )

                image_features = mobilevit_model(images)
                text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                text_features = text_outputs[:, 0, :]

                outputs = classifier(image_features, text_features).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_accuracy += compute_accuracy(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        print(f"Validation completed. Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}")

        # Optionally clear GPU cache
        torch.cuda.empty_cache()

    print("Saving model...")
    torch.save(classifier.state_dict(), "clip_classifier_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    main()
