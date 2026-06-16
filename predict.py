import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
from dataset import KidneyDataset
import pandas as pd
import json

# 1. ПУТИ И УСТРОЙСТВО
DATA_DIR = "./data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. ПОДГОТОВКА ВАЛИДАЦИОННЫХ ID
tile_meta = pd.read_csv(os.path.join(DATA_DIR, "tile_meta.csv"))
annotated_ids = set()
with open(os.path.join(DATA_DIR, "polygons.jsonl"), 'r') as f:
    for line in f:
        data = json.loads(line)
        annotated_ids.add(data['id'])

annotated_meta = tile_meta[tile_meta['id'].isin(annotated_ids)].copy()
# Наша валидационная выборка - только WSI 4
val_ids = annotated_meta[annotated_meta['source_wsi'] == 4]['id'].tolist()


# 3. ЗАГРУЗКА МОДЕЛИ И ВЕСОВ
print("Загрузка модели U-Net...")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None, # Веса мы загрузим свои собственные
    in_channels=3,
    classes=1,
    activation=None
)
# Загружаем сохраненные веса лучшей эпохи
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()
print("Модель успешно загружена!")


# 4. СОЗДАЕМ ВАЛИДАЦИОННЫЙ ДАТАСЕТ
val_dataset = KidneyDataset(data_dir=DATA_DIR, image_ids=val_ids, transform=None)


# 5. ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ
# Выберем 3 случайных индекса из валидационного датасета
random_indices = random.sample(range(len(val_dataset)), 3)

for i, idx in enumerate(random_indices):
    img_tensor, mask_tensor = val_dataset[idx]
    
    # Добавляем размерность батча и переносим на видеокарту
    input_batch = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Получаем предсказание (логиты)
        prediction = model(input_batch)
        # Пропускаем через Sigmoid, чтобы получить вероятности от 0 до 1
        prediction_probs = torch.sigmoid(prediction)
        # Применяем порог 0.5 (все, что больше 50% вероятности - сосуд, иначе - фон)
        pred_mask = (prediction_probs > 0.5).float()
    
    # Переводим тензоры обратно в numpy массивы для отрисовки
    img_show = img_tensor.permute(1, 2, 0).numpy()
    true_mask_show = mask_tensor.squeeze().numpy()
    pred_mask_show = pred_mask.squeeze().cpu().numpy()
    
    # Рисуем три картинки рядом
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_show)
    axes[0].set_title(f"Снимок ткани (Индекс: {idx})")
    axes[0].axis('off')
    
    axes[1].imshow(true_mask_show, cmap='gray')
    axes[1].set_title("Разметка врача (Ground Truth)")
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask_show, cmap='gray')
    axes[2].set_title("Предсказание нашей нейросети")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()