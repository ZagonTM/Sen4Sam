import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Importation des modules locaux
from src.models.sam2_decoder_adapter import build_adapted_sam2
from src.data.sentinel_dataset import Sentinel2Dataset
from src.utils.losses import CombinedLoss
from src.utils.evaluation_metrics import calculate_iou

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_model(config_path="configs/ablation2_finetune_64.yaml"):
    config = load_config(config_path)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    # 1. Initialisation
    model = build_adapted_sam2(
        config_file=config['model']['config_file'],
        ckpt_path=config['model']['checkpoint_path'],
        device=device
    )

    dataset = Sentinel2Dataset(
        image_dir=config['data']['image_dir'],
        mask_dir=config['data']['mask_dir'],
        grid_size=config['data']['grid_size'],
        target_size=config['data']['target_size']
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    criterion = CombinedLoss()

    # 2. Paramètres d'Early Stopping (Basé sur l'IoU)
    best_val_iou = 0.0  # Initialisation à 0 (on cherche à maximiser)
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    best_model_path = os.path.join(config['training']['save_dir'], "best_sam2_finetuned.pt")

    # 3. Boucle d'entraînement
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in loop:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            points, labels = batch['points'].to(device), batch['point_labels'].to(device)

            optimizer.zero_grad()

            image_embeddings = model.image_encoder(images)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(points, labels), boxes=None, masks=None
            )
            predicted_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            loss = criterion(predicted_masks, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 4. Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, masks = batch['image'].to(device), batch['mask'].to(device)
                points, labels = batch['points'].to(device), batch['point_labels'].to(device)

                image_embeddings = model.image_encoder(images)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=(points, labels), boxes=None, masks=None
                )
                predicted_masks, _ = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                loss = criterion(predicted_masks, masks)
                val_loss += loss.item()
                val_iou += calculate_iou(predicted_masks, masks)  # Calcul de l'IoU

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # 5. Early Stopping (Maximisation de l'IoU)
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Nouveau meilleur modèle sauvegardé (IoU: {best_val_iou:.4f}) : {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping déclenché après {epoch+1} époques. Meilleur IoU : {best_val_iou:.4f}")
                break

if __name__ == "__main__":
    train_model()