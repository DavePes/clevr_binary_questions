import os
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from timm import create_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class RandomQADataset(Dataset):
    """
    Loads images from one .npz (with key 'images') and a separate .npz for Q&A (with default key 'arr_0').
    For each image index, picks a random (question, answer) pair. 
    `answer` is turned into float 0.0 or 1.0.
    """
    def __init__(self, images_path, qa_path):
        # Load images (mmap_mode to avoid loading all at once)
        #   images.npz -> key 'images'
        images_data = np.load(images_path, mmap_mode='r')
        self.images = images_data['images']  # shape: (N, H, W, C), dtype=uint8

        # Load Q&A object array
        #   ql.npz -> key 'arr_0', shape: [N], each item an array of shape (K_i, 2)
        qa_data = np.load(qa_path, mmap_mode='r', allow_pickle=True)
        self.qa_list = qa_data['arr_0']  # shape: [N], each item = np.array([...], dtype=object)

        # Make sure we have the same number of images and Q&A entries
        assert len(self.images) == len(self.qa_list), \
            f"Mismatch: images={len(self.images)}, qa={len(self.qa_list)}"

        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1) Load image
        image_uint8 = self.images[idx]  # shape: (H, W, C)

        # 2) Convert to float32 and normalize
        image_tensor = torch.tensor(image_uint8, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # shape: (3, H, W)

        # 3) Retrieve all QA pairs for this index
        #    Suppose shape: (K_i, 2), each row = [question_str, answer_bool]
        qa_array = self.qa_list[idx]
        # pick one random row
        row_idx = random.randint(0, len(qa_array) - 1)
        question_str, answer_bool = qa_array[row_idx]

        label = torch.tensor(answer_bool, dtype=torch.float32)

        # Return raw question_str for tokenization in collate_fn
        return (image_tensor, question_str, label)

def collate_fn(batch, tokenizer=None):
    """
    Custom collate function for batch-level tokenization.
    batch: list of (image_tensor, question_str, label)
    """
    images, question_strs, labels = zip(*batch)

    # Stack images => [batch_size, 3, H, W]
    images = torch.stack(images)
    labels = torch.stack(labels)

    # Tokenize questions
    encoded = tokenizer(
        list(question_strs),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]           # shape: [batch_size, seq_len]
    attention_mask = encoded["attention_mask"] # shape: [batch_size, seq_len]

    return images, input_ids, attention_mask, labels

# Classifier that merges image + text embeddings
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ### Hyperparameters ###
    batch_size = 64
    num_workers = 4
    num_epochs = 5
    lr_classifier = 1e-4
    lr_bert = 2e-5
    lr_mobilevit = 1e-5
    #######################

    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 2) Create & load models
    bert_model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    mobilevit_model = create_model('mobilevit_s', pretrained=True)
    # Remove classifier head to get the embeddings
    mobilevit_model.classifier = nn.Identity()
    mobilevit_model.to(device)

    # 3) Create Datasets (images + QA)
    #    Adjust file paths to your own structure
    train_images_path = "raw/images_train.npz"
    train_qa_path = "raw/ql_train.npz"
    val_images_path = "raw/images_val.npz"
    val_qa_path = "raw/ql_val.npz"

    train_dataset = RandomQADataset(train_images_path, train_qa_path)
    val_dataset = RandomQADataset(val_images_path, val_qa_path)

    # 4) Create DataLoaders (with custom collate_fn)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer)
    )

    # 5) Figure out image feature dimension by passing a dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_output = mobilevit_model(dummy_input)  # shape [1, features]
    image_dim = dummy_output.shape[1]
    print(f"MobileViT output dimension: {image_dim}")

    # 6) Figure out text dimension from BERT config
    text_dim = bert_model.config.hidden_size
    print(f"DistilBERT hidden size: {text_dim}")

    # 7) Initialize our final classifier
    classifier = Classifier(image_dim, text_dim).to(device)

    # 8) Define optimizers & loss
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr_classifier)
    bert_optimizer = optim.Adam(bert_model.parameters(), lr=lr_bert)
    mobilevit_optimizer = optim.Adam(mobilevit_model.parameters(), lr=lr_mobilevit)
    criterion = nn.BCELoss()

    print("Setup complete. Training...")

    # 9) Training loop
    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        classifier.train()
        bert_model.train()
        mobilevit_model.train()

        total_loss = 0.0
        total_accuracy = 0.0

        for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc="Train")):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Zero gradients
            classifier_optimizer.zero_grad()
            bert_optimizer.zero_grad()
            mobilevit_optimizer.zero_grad()

            # Forward pass
            image_features = mobilevit_model(images)  # shape: [batch_size, image_dim]
            text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            text_features = text_outputs[:, 0, :]  # [CLS] embedding from DistilBERT

            outputs = classifier(image_features, text_features).squeeze()
            loss = criterion(outputs, labels)

            # Backprop
            loss.backward()
            classifier_optimizer.step()
            bert_optimizer.step()
            mobilevit_optimizer.step()

            # Logging
            total_loss += loss.item()
            total_accuracy += compute_accuracy(outputs, labels).item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_accuracy / len(train_loader)
        print(f"Train - Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

        # Validation
        classifier.eval()
        bert_model.eval()
        mobilevit_model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(tqdm(val_loader, desc="Val")):
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                image_features = mobilevit_model(images)
                text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                text_features = text_outputs[:, 0, :]

                outputs = classifier(image_features, text_features).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += compute_accuracy(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        print(f"Val - Epoch {epoch+1}: Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f}")
        torch.cuda.empty_cache()

    print("Saving model...")
    torch.save(classifier.state_dict(), "clip_classifier_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()
