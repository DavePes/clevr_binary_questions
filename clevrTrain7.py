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
    Loads images from one .npz (with key 'images') and a separate .npz for Q&A.
    For each image index, picks a random (question, answer) pair. 
    """
    def __init__(self, images_path, qa_path):
        images_data = np.load(images_path, mmap_mode='r')
        self.images = images_data['images']  # shape: (N, H, W, C), dtype=uint8

        qa_data = np.load(qa_path, mmap_mode='r', allow_pickle=True)
        self.qa_list = qa_data['arr_0']  # shape: [N], each item = (K_i, 2)

        assert len(self.images) == len(self.qa_list), \
            f"Mismatch: images={len(self.images)}, qa={len(self.qa_list)}"

        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_uint8 = self.images[idx]
        image_tensor = torch.tensor(image_uint8, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # shape: (3, H, W)

        qa_array = self.qa_list[idx]
        row_idx = random.randint(0, len(qa_array) - 1)
        question_str, answer_bool = qa_array[row_idx]
        label_value = 1.0 if answer_bool == "True" else 0.0

        label = torch.tensor(label_value, dtype=torch.float32)
        return (image_tensor, question_str, label)


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


class MultiScaleCrossModalTransformer(nn.Module):
    """
    Collects multiple feature maps from MobileViT, projects each to hidden_dim,
    flattens, concatenates them into a single token set, then fuses with text.
    """
    def __init__(self, 
                 in_channels_list,  # e.g. [96, 128, 640]
                 text_in_dim,      # e.g. 768 for DistilBERT
                 hidden_dim=256, 
                 nheads=4, 
                 num_layers=2, 
                 dropout=0.1):
        super().__init__()

        # A projection for each feature map scale
        # We'll store small Conv2d( in_c, hidden_dim, kernel_size=1 ) for each scale
        self.projections = nn.ModuleList([
            nn.Conv2d(in_c, hidden_dim, kernel_size=1) for in_c in in_channels_list
        ])

        # Project text dimension -> hidden_dim
        self.text_projection = nn.Linear(text_in_dim, hidden_dim)

        # A standard TransformerEncoder for cross-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=4*hidden_dim, 
            dropout=dropout,
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, feature_maps_list, text_embeds):
        """
        feature_maps_list: list of feature maps from MobileViT
          - e.g. [ (B, C1, H1, W1), (B, C2, H2, W2), (B, C3, H3, W3) ]
        text_embeds: [B, seq_len, text_in_dim] from BERT
        """
        B = text_embeds.size(0)

        all_img_tokens = []

        # For each scale's feature map, project -> hidden_dim, flatten -> tokens
        for proj, fm in zip(self.projections, feature_maps_list):
            # fm shape: (B, C, H, W)
            fm_proj = proj(fm)  # -> (B, hidden_dim, H, W)
            fm_proj = fm_proj.flatten(2)  # -> (B, hidden_dim, H*W)
            fm_proj = fm_proj.permute(0, 2, 1)  # -> (B, H*W, hidden_dim)
            all_img_tokens.append(fm_proj)

        # Concatenate all scales along the token dimension
        # If we have 3 scales, we might get (B, N1+N2+N3, hidden_dim)
        img_tokens = torch.cat(all_img_tokens, dim=1)

        # Project text -> hidden_dim
        text_tokens = self.text_projection(text_embeds)  # (B, seq_len, hidden_dim)

        # Combine image + text tokens
        combined = torch.cat([img_tokens, text_tokens], dim=1)  # (B, N_total + seq_len, hidden_dim)

        # Run through transformer
        fused = self.transformer_encoder(combined)  # (B, N_combined, hidden_dim)

        # Pool (mean) over all tokens
        fused_mean = fused.mean(dim=1)  
        logits = self.classifier(fused_mean)
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
    batch_size = 32
    num_workers = 4
    num_epochs = 10
    lr_transformer = 1e-4
    lr_bert = 2e-5
    lr_mobilevit = 2e-5
    #######################

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # We want multiple feature maps => out_indices=[2, 3, 4], for example
    mobilevit_model = create_model(
        'mobilevit_s',
        pretrained=True,
        features_only=True,
        out_indices=[2,3,4]
    ).to(device)

    bert_model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)

    train_images_path = "raw/train/images.npz"
    train_qa_path = "raw/train/ql.npz"
    val_images_path = "raw/val/images.npz"
    val_qa_path = "raw/val/ql.npz"

    train_dataset = RandomQADataset(train_images_path, train_qa_path)
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

    # Check shapes of the multiple feature maps
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_outs = mobilevit_model(dummy_input)  # list of 3 feature maps
    for i, fm in enumerate(dummy_outs):
        print(f"Feature map {i} shape: {fm.shape}")
        # e.g. [1, 96, 28, 28], [1, 128, 14, 14], [1, 640, 7, 7]

    in_channels_list = [fm.shape[1] for fm in dummy_outs]
    print("in_channels_list =", in_channels_list)

    text_dim = bert_model.config.hidden_size  
    print(f"DistilBERT hidden size: {text_dim}")

    # Initialize multi-scale cross-modal transformer
    cross_modal_model = MultiScaleCrossModalTransformer(
        in_channels_list=in_channels_list,  # e.g. [96, 128, 640]
        text_in_dim=text_dim,
        hidden_dim=256,
        nheads=4,
        num_layers=2,
        dropout=0.25
    ).to(device)

    optimizer = optim.AdamW([
        {'params': cross_modal_model.parameters(), 'lr': lr_transformer},
        {'params': bert_model.parameters(),         'lr': lr_bert},
        {'params': mobilevit_model.parameters(),    'lr': lr_mobilevit},
    ])
    criterion = nn.BCELoss()

    print("Setup complete. Training...")

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

            optimizer.zero_grad()

            # Get multiple feature maps
            # image_features_list is a list: [fm2, fm3, fm4]
            image_features_list = mobilevit_model(images)

            # BERT text features => [B, seq_len, 768]
            text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            # Forward in our multi-scale transformer
            outputs = cross_modal_model(image_features_list, text_outputs).squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
                text_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

                outputs = cross_modal_model(image_features_list, text_outputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += compute_accuracy(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        print(f"Val - Epoch {epoch+1}: Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f}")
        torch.cuda.empty_cache()

    print("Saving model...")
    torch.save(cross_modal_model.state_dict(), "clevr_cross_attn_multiscale.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()
