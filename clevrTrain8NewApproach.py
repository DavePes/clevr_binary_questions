import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from timm import create_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


############################################
# 1. Dataset with Basic Augmentation
############################################

class RandomQADataset(Dataset):
    """
    Loads images from one .npz (with key 'images') and a separate .npz for Q&A.
    For each image index, picks a random (question, answer) pair.

    We apply standard image augmentations to each image, ensuring the final
    spatial resolution is 224 x 224.
    """
    def __init__(self, images_path, qa_path, transform=None):
        images_data = np.load(images_path, mmap_mode='r')
        self.images = images_data['images']  # shape: (N, H, W, C), dtype=uint8

        qa_data = np.load(qa_path, mmap_mode='r', allow_pickle=True)
        self.qa_list = qa_data['arr_0']  # shape: [N], each item = list of (question, "True"/"False")

        assert len(self.images) == len(self.qa_list), \
            f"Mismatch: images={len(self.images)}, qa={len(self.qa_list)}"

        self.length = len(self.images)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_uint8 = self.images[idx]  # (H, W, C), dtype=uint8

        # Convert to PIL so we can use torchvision transforms
        pil_image = Image.fromarray(image_uint8)

        if self.transform is not None:
            image_tensor = self.transform(pil_image) # (3, 224, 224)
        else:
            # Fallback: just convert to tensor 0..1 if no transform
            image_tensor = torch.tensor(image_uint8, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Pick a random question
        qa_array = self.qa_list[idx]
        row_idx = random.randint(0, len(qa_array) - 1)
        question_str, answer_bool = qa_array[row_idx]
        label_value = 1.0 if answer_bool == "True" else 0.0
        label = torch.tensor(label_value, dtype=torch.float32)

        return (image_tensor, question_str, label)


############################################
# 2. Collate Function for BERT Tokenization
############################################

def collate_fn(batch, tokenizer=None):
    images, question_strs, labels = zip(*batch)

    images = torch.stack(images)
    labels = torch.stack(labels)

    encoded = tokenizer(
        list(question_strs),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    return images, input_ids, attention_mask, labels


############################################
# 3. Simple Cross-Modal Model
#    - single feature map from EfficientNet
#    - mean-pool text from DistilBERT
#    - fuse with small MLP
############################################

class SimpleCrossModalModel(nn.Module):
    def __init__(self, vision_dim, text_dim, hidden_dim=256,dropout_rate=0.4):
        super().__init__()
        self.vision_fc = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Then maybe:
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2*hidden_dim, 1)
)

    def forward(self, vision_feats, text_feats):
        """
        vision_feats: (B, vision_dim) after global pooling
        text_feats: (B, seq_len, text_dim)
        """
        # Mean-pool over the text sequence dimension
        text_feats = text_feats.mean(dim=1)  # (B, text_dim)

        # Project
        v_proj = self.vision_fc(vision_feats)
        t_proj = self.text_fc(text_feats)

        # Simple cat
        fused = torch.cat([v_proj, t_proj], dim=1)  # (B, 2*hidden_dim)

        # Classify (binary)
        logits = self.classifier(fused)  # (B, 1)
        return torch.sigmoid(logits)


def compute_accuracy(predictions, labels):
    predicted_labels = (predictions >= 0.5).float()
    correct = (predicted_labels == labels).float()
    accuracy = correct.sum() / correct.size(0)
    return accuracy


############################################
# 4. Main Training Loop
############################################

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------------------
    # Hyperparameters
    # ---------------------
    batch_size = 64
    num_workers = 4
    num_epochs = 10

    # Learning rates for the different parts of the model
    lr_vision_backbone = 2e-5  # smaller LR for pretrained backbone
    lr_bert = 2e-5             # smaller LR for BERT
    lr_head = 1e-4             # higher LR for new cross-modal head

    # ---------------------
    # Data Augmentation
    # ---------------------
    # Make sure final size is 224x224
    train_transforms = T.Compose([
        T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
        T.RandomRotation(degrees=4),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.RandomResizedCrop(224, scale=(0.9, 1.1)),
        T.ToTensor(),
    ])
    # val_transforms = T.Compose([
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    # ])

    # ---------------------
    # Prepare Dataset & DataLoader
    # ---------------------
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_images_path = "raw/train/images.npz"
    train_qa_path = "raw/train/ql.npz"
    val_images_path = "raw/val/images.npz"
    val_qa_path = "raw/val/ql.npz"

    train_dataset = RandomQADataset(train_images_path, train_qa_path, transform=train_transforms)
    val_dataset = RandomQADataset(val_images_path, val_qa_path)

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

    # ---------------------
    # Vision Backbone: EfficientNet-B1
    # features_only=True => returns intermediate feature maps
    # but we'll just take the last one: out_indices=[4]
    # (the final layer before the classifier).
    # ---------------------
    effnet_model = create_model(
        'efficientnet_b1',  # ~7.8M params
        pretrained=True,
        features_only=True,
        out_indices=[4]     # only the final feature map
    ).to(device)

    # Check the output dimension of the final feature map
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_outs = effnet_model(dummy_input)  # list of length=1
    final_fm = dummy_outs[0]  # shape: (1, C, H, W)
    print("Final feature map shape from EfficientNet-B1:", final_fm.shape)
    # Typically (1, 1280, 7, 7) for B1 => 1280 channels

    vision_dim = final_fm.shape[1]

    # ---------------------
    # Text Model: DistilBERT
    # ---------------------
    bert_model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    text_dim = bert_model.config.hidden_size  # 768

    # ---------------------
    # Cross-Modal Head
    # ---------------------
    cross_modal_model = SimpleCrossModalModel(
        vision_dim=vision_dim,
        text_dim=text_dim,
        hidden_dim=256
    ).to(device)

    # ---------------------
    # Optimizer
    # ---------------------
    # We use separate parameter groups so we can set different LRs.
    optimizer = optim.AdamW([
        {'params': effnet_model.parameters(),      'lr': lr_vision_backbone},
        {'params': bert_model.parameters(),        'lr': lr_bert},
        {'params': cross_modal_model.parameters(), 'lr': lr_head}
    ])
    criterion = nn.BCELoss()

    # ---------------------
    # Training Loop
    # ---------------------
    print("Setup complete. Training...")

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        effnet_model.train()
        bert_model.train()
        cross_modal_model.train()

        total_loss = 0.0
        total_acc = 0.0

        for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc="Train")):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 1) Vision forward: single final feature map
            # shape = (B, C, H, W)
            fm = effnet_model(images)[0]

            # Global average pool to get (B, C)
            # or flatten if you prefer
            B, C, H, W = fm.shape
            vision_feats = fm.view(B, C, H*W).mean(dim=2)  # (B, C)

            # 2) Text forward: DistilBERT => last_hidden_state => (B, seq_len, 768)
            text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state  # (B, seq_len, 768)

            # 3) Cross-modal forward
            outputs = cross_modal_model(vision_feats, text_embeds).squeeze()  # (B,)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += compute_accuracy(outputs, labels).item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"Train - Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

        # ---------------------
        # Validation
        # ---------------------
        effnet_model.eval()
        bert_model.eval()
        cross_modal_model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(tqdm(val_loader, desc="Val")):
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                fm = effnet_model(images)[0]
                B, C, H, W = fm.shape
                vision_feats = fm.view(B, C, H*W).mean(dim=2)

                text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                text_embeds = text_outputs.last_hidden_state

                outputs = cross_modal_model(vision_feats, text_embeds).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += compute_accuracy(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        print(f"Val - Epoch {epoch+1}: Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f}")
        torch.cuda.empty_cache()

    # ---------------------
    # Save model
    # ---------------------
    print("Saving model...")
    torch.save(cross_modal_model.state_dict(), "clevr_simple_effnet_b1.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()
