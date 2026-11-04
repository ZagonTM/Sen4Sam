import torch
import torch.nn as nn
from sam2.build_sam import build_sam2

def build_adapted_sam2(config_file="sam2_hiera_s.yaml", ckpt_path="checkpoints/sam2_hiera_small.pt", device="cuda"):
    """
    Charge SAM 2.1, gèle l'Image Encoder et le Prompt Encoder,
    et rend entraînables les couches d'attention du Mask Decoder.
    """
    # 1. Chargement du modèle de base
    model = build_sam2(config_file, ckpt_path, device=device)
    model.train() # Mode entraînement activé

    # 2. Gel de l'Image Encoder (aucun apprentissage sur l'extraction visuelle)
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # 3. Gel du Prompt Encoder (les embeddings de points/boîtes restent standards)
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False

    # 4. Préparation du Mask Decoder : tout geler d'abord
    for param in model.mask_decoder.parameters():
        param.requires_grad = False

    # 5. Dégel exclusif des couches d'auto-attention et cross-attention
    trainable_params = 0
    for name, param in model.mask_decoder.named_parameters():
        if "attn" in name or "attention" in name or "cross_attn" in name:
            param.requires_grad = True
            trainable_params += param.numel()

    print(f"Modèle adapté avec succès. Paramètres entraînables du Mask Decoder : {trainable_params:,}")

    return model

if __name__ == "__main__":
    # Test unitaire rapide pour vérifier l'adaptation
    # Nécessite d'avoir les fichiers YAML et PT de SAM 2.1 accessibles
    try:
        adapted_sam = build_adapted_sam2(device="cpu")

        # Vérification des gradients
        image_encoder_frozen = all(not p.requires_grad for p in adapted_sam.image_encoder.parameters())
        print(f"Image Encoder entièrement gelé : {image_encoder_frozen}")

    except FileNotFoundError:
        print("Erreur : Fichiers de configuration ou poids SAM 2.1 introuvables. Vérifiez les chemins d'accès.")
