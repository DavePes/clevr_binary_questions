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
        # Load images
        images_data = np.load(images_path, mmap_mode='r')
        self.images = images_data['images']  # shape: (N, H, W, C), dtype=uint8

        # Load Q&A
        qa_data = np.load(qa_path, mmap_mode='r', allow_pickle=True)
        self.qa_list = qa_data['arr_0']  # shape: [N], each item = np.array([...], dtype=object)

        # Check
        assert len(self.images) == len(self.qa_list), \
            f"Mismatch: images={len(self.images)}, qa={len(self.qa_list)}"

        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1) Load image
        image_uint8 = self.images[idx]  # shape: (H, W, C)
        image_tensor = torch.tensor(image_uint8, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # shape: (3, H, W)

        # 2) Pick a random QA pair for this image
        qa_array = self.qa_list[idx]  # shape (K_i, 2)
        row_idx = random.randint(0, len(qa_array) - 1)
        question_str, answer_bool = qa_array[row_idx]
        if answer_bool == "True":
            label_value = 1.0
        else:
            label_value = 0.0

        label = torch.tensor(label_value, dtype=torch.float32)

        return (image_tensor, question_str, label)


def collate_fn(batch, tokenizer=None):
    """
    Custom collate function for batch-level tokenization.
    batch: list of (image_tensor, question_str, label)
    """
    images, question_strs, labels = zip(*batch)

    # Stack images => [batch_size, 3, H, W]
    images = torch.stack(images)
    # For MobileViT from timm, it actually expects (B, 3, H, W) in RGB order by default.
    # If you have a model that expects BGR, you might reorder channels here.
    #images = images[:, [2, 1, 0], :, :]  # Uncomment if truly needed

    labels = torch.stack(labels)

    # Tokenize questions
    encoded = tokenizer(
        list(question_strs),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]           # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"] # [batch_size, seq_len]

    return images, input_ids, attention_mask, labels


class CrossModalTransformer(nn.Module):
    """
    A small transformer-based fusion module:
      1) Projects the image feature map -> hidden_dim
      2) Projects the text embeddings -> hidden_dim
      3) Concat the image tokens + text tokens
      4) Run multiple layers of self-attention (TransformerEncoder)
      5) Pool the final fused tokens + produce a single sigmoid output
    """
    def __init__(self, img_in_channels, text_in_dim, hidden_dim=256, nheads=4, num_layers=2, dropout=0.1):
        super().__init__()

        # Project from image channels -> hidden_dim
        self.image_projection = nn.Conv2d(img_in_channels, hidden_dim, kernel_size=1)

        # Project text dimension -> same hidden_dim
        self.text_projection = nn.Linear(text_in_dim, hidden_dim)

        # Define a standard TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=4*hidden_dim,  # typical 4x the model dim
            dropout=dropout,
            batch_first=True  # so the input shape is (B, seq_len, hidden_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier after we pool the tokens
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, image_map, text_embeds):
        """
        image_map: [B, C, H, W] from MobileViT final stage 
        text_embeds: [B, seq_len, text_in_dim] from BERT last_hidden_state
        """
        B, C, H, W = image_map.shape

        # 1) Project the image channels to hidden_dim
        image_map = self.image_projection(image_map)  # [B, hidden_dim, H, W]

        # 2) Flatten spatial dimension => (B, H*W, hidden_dim)
        image_map = image_map.flatten(2)  # shape => [B, hidden_dim, H*W]
        image_map = image_map.permute(0, 2, 1)  # => [B, H*W, hidden_dim]

        # 3) Project text from BERT dimension -> hidden_dim
        text_tokens = self.text_projection(text_embeds)  # [B, seq_len, hidden_dim]

        # 4) Concat image tokens + text tokens => (B, N+T, hidden_dim)
        combined = torch.cat([image_map, text_tokens], dim=1)

        # 5) Pass through the TransformerEncoder
        fused = self.transformer_encoder(combined)  # [B, N+T, hidden_dim]

        # 6) Pool the fused tokens (mean-pooling or a [CLS]-like approach)
        fused_mean = fused.mean(dim=1)  # [B, hidden_dim]

        # 7) Binary classification
        logits = self.classifier(fused_mean)  # [B, 1]
        return torch.sigmoid(logits)


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
    num_epochs = 10
    lr_transformer = 1e-4
    lr_bert = 1e-5
    lr_mobilevit = 1e-5
    #######################

    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 2) Create & load models
    #    We use features_only=True so MobileViT returns intermediate feature maps
    #    out_indices=[4] => the final stage
    mobilevit_model = create_model(
        'mobilevit_s',
        pretrained=True,
        features_only=True,
        out_indices=[4]  # the last feature map: shape (B, 640, 7, 7)
    ).to(device)

    # DistilBERT for text
    bert_model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    
    # 3) Create Datasets
    train_images_path = "raw/train/images.npz"
    train_qa_path = "raw/train/ql.npz"
    val_images_path = "raw/val/images.npz"
    val_qa_path = "raw/val/ql.npz"

    train_dataset = RandomQADataset(train_images_path, train_qa_path)
    val_dataset = RandomQADataset(val_images_path, val_qa_path)

    # 4) Create DataLoaders
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

    # 5) Figure out the shape of the final MobileViT feature map
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_output_list = mobilevit_model(dummy_input)  
    # We used out_indices=[4], so dummy_output_list is a list of length 1
    dummy_output = dummy_output_list[0]  # shape => [1, 640, 7, 7]
    print("MobileViT final feature map shape:", dummy_output.shape)
    img_in_channels = dummy_output.shape[1]  # e.g. 640

    # 6) Text dimension from DistilBERT
    text_dim = bert_model.config.hidden_size  # e.g. 768
    print(f"DistilBERT hidden size: {text_dim}")

    # 7) Initialize our cross-modal transformer
    #    You can tune hidden_dim, nheads, num_layers as you like
    cross_modal_model = CrossModalTransformer(
        img_in_channels=img_in_channels,
        text_in_dim=text_dim,
        hidden_dim=256,   # or 512, etc.
        nheads=4,
        num_layers=2,
        dropout=0.1
    ).to(device)

    # 8) Define optimizers & loss
    #    We'll train: cross_modal_model, bert_model, and the MobileViT parameters
    optimizer = optim.Adam([
        {'params': cross_modal_model.parameters(), 'lr': lr_transformer},
        {'params': bert_model.parameters(),         'lr': lr_bert},
        {'params': mobilevit_model.parameters(),    'lr': lr_mobilevit},
    ])
    criterion = nn.BCELoss()

    print("Setup complete. Training...")

    # 9) Training loop
    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        cross_modal_model.train()
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
            optimizer.zero_grad()

            # Forward pass
            # (a) Get image feature map from MobileViT
            image_features_list = mobilevit_model(images)  
            # out_indices=[4] => a single-element list
            image_map = image_features_list[0]  # [B, 640, 7, 7]

            # (b) Get full BERT embeddings (or you can just do last_hidden_state[:,0,:])
            text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # shape => [B, seq_len, 768]

            # (c) Fuse via cross-modal transformer
            outputs = cross_modal_model(image_map, text_outputs).squeeze()  # [B]
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Logging
            total_loss += loss.item()
            total_accuracy += compute_accuracy(outputs, labels).item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_accuracy / len(train_loader)
        print(f"Train - Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

        # Validation
        cross_modal_model.eval()
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

                image_features_list = mobilevit_model(images)
                image_map = image_features_list[0]
                text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

                outputs = cross_modal_model(image_map, text_outputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += compute_accuracy(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        print(f"Val - Epoch {epoch+1}: Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f}")
        torch.cuda.empty_cache()

    print("Saving model...")
    torch.save(cross_modal_model.state_dict(), "clevr_cross_attn_model.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()
