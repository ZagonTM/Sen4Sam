import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importation des modules locaux
from src.models.sam2_decoder_adapter import build_adapted_sam2
from sam2.build_sam import build_sam2
from src.data.sentinel_dataset import Sentinel2Dataset
from src.utils.evaluation_metrics import calculate_iou, PerformanceProfiler

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def evaluate_configuration(config_path, is_finetuned=False, test_dir="data/sentinel/test"):
    config = load_config(config_path)
    device = torch.device(config.get('data', {}).get('device', 'cuda') if torch.cuda.is_available() else "cpu")

    # 1. Chargement du modèle approprié
    if is_finetuned:
        model = build_adapted_sam2(
            config_file=config['model']['config_file'],
            ckpt_path=config['model']['checkpoint_path'],
            device=device
        )
        # Charger les poids optimisés du Mask Decoder
        finetuned_ckpt = os.path.join(config['training']['save_dir'], "best_sam2_finetuned.pt")
        model.load_state_dict(torch.load(finetuned_ckpt, map_location=device))
    else:
        model = build_sam2(
            config_file=config['model']['config_file'],
            ckpt_path=config['model']['checkpoint_path'],
            device=device
        )

    model.eval()

    # 2. Préparation des données de test
    dataset = Sentinel2Dataset(
        image_dir=os.path.join(test_dir, "images"),
        mask_dir=os.path.join(test_dir, "masks"),
        grid_size=config['data']['grid_size'],
        target_size=config['data']['target_size']
    )
    # Batch size de 1 pour mesurer précisément la latence d'inférence unitaire
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_iou = 0.0
    total_fps = 0.0

    # 3. Boucle d'évaluation avec profilage matériel
    with torch.no_grad():
        with PerformanceProfiler(device=device.type) as profiler:
            for batch in tqdm(dataloader, desc=f"Evaluation {config['experiment']['name']}"):
                images, masks = batch['image'].to(device), batch['mask'].to(device)
                points, labels = batch['points'].to(device), batch['point_labels'].to(device)

                # Mesure de latence individuelle par image
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()

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

                end_event.record()
                torch.cuda.synchronize()

                # Calcul de l'IoU
                iou = calculate_iou(predicted_masks, masks)
                total_iou += iou

                # Calcul FPS (1000 ms / temps d'exécution en ms)
                latency_ms = start_event.elapsed_time(end_event)
                total_fps += (1000.0 / latency_ms) if latency_ms > 0 else 0

    # 4. Agrégation des métriques
    avg_iou = total_iou / len(dataloader)
    avg_fps = total_fps / len(dataloader)
    peak_vram = profiler.get_peak_vram_mb()

    return {
        "Configuration": config['experiment']['name'],
        "Grid Size": config['data']['grid_size'],
        "IoU Moyen": avg_iou,
        "FPS Moyen": avg_fps,
        "VRAM Max (MB)": peak_vram
    }

if __name__ == "__main__":
    # Assurez-vous d'avoir un dossier de test (ex: data/sentinel/test/images et masks)
    test_directory = "data/sentinel/test"

    print("Démarrage de l'évaluation des ablations...\n")

    results = []

    # Évaluation 1 : Baseline (Zero-shot, grille 32x32)
    results.append(evaluate_configuration("configs/baseline_32.yaml", is_finetuned=False, test_dir=test_directory))

    # Évaluation 2 : Ablation 1 (Zero-shot, grille 64x64)
    results.append(evaluate_configuration("configs/ablation1_64.yaml", is_finetuned=False, test_dir=test_directory))

    # Évaluation 3 : Ablation 2 (Fine-tuning Mask Decoder, grille 64x64)
    results.append(evaluate_configuration("configs/ablation2_finetune_64.yaml", is_finetuned=True, test_dir=test_directory))

    # Affichage des résultats
    print("\n" + "="*70)
    print(f"{'Configuration':<25} | {'Grid':<5} | {'IoU (%)':<8} | {'FPS':<6} | {'VRAM (MB)':<10}")
    print("="*70)
    for res in results:
        print(f"{res['Configuration']:<25} | {res['Grid Size']:<5} | {res['IoU Moyen']*100:>6.2f}% | {res['FPS Moyen']:>6.1f} | {res['VRAM Max (MB)']:>8.1f}")
    print("="*70)