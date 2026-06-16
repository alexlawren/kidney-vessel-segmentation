import os
import json
import pandas as pd
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from dataset import KidneyDataset

# 1. ОПРЕДЕЛЯЕМ ПУТИ И СПИСКИ ID
DATA_DIR = "./data"
tile_meta = pd.read_csv(os.path.join(DATA_DIR, "tile_meta.csv"))

# Получаем список только размеченных ID
annotated_ids = set()
with open(os.path.join(DATA_DIR, "polygons.jsonl"), 'r') as f:
    for line in f:
        data = json.loads(line)
        annotated_ids.add(data['id'])

# Фильтруем метаданные
annotated_meta = tile_meta[tile_meta['id'].isin(annotated_ids)].copy()

# Честное разделение: Train - WSI 1, 2, 3; Val - WSI 4
train_ids = annotated_meta[annotated_meta['source_wsi'].isin([1, 2, 3])]['id'].tolist()
val_ids = annotated_meta[annotated_meta['source_wsi'] == 4]['id'].tolist()

print(f"Изображений для обучения (Train): {len(train_ids)}")
print(f"Изображений для проверки (Val): {len(val_ids)}")


# 2. НАСТРАИВАЕМ АУГМЕНТАЦИИ (ALBUMENTATIONS)
# Сильные геометрические повороты, как у победителя соревнования
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=180, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# Для валидации аугментации не нужны, только исходные картинки
val_transform = None


# 3. СОЗДАЕМ ДАТАСЕТЫ
train_dataset = KidneyDataset(data_dir=DATA_DIR, image_ids=train_ids, transform=train_transform)
val_dataset = KidneyDataset(data_dir=DATA_DIR, image_ids=val_ids, transform=val_transform)


# 4. ИНИЦИАЛИЗИРУЕМ МОДЕЛЬ U-NET (БИБЛИОТЕКА SMP)
print("Инициализация модели U-Net...")
model = smp.Unet(
    encoder_name="resnet34",        # Быстрый и мощный энкодер
    encoder_weights="imagenet",     # Предобученные веса ImageNet
    in_channels=3,                  # RGB картинка на входе
    classes=1,                      # Бинарная маска на выходе (1 класс - сосуды)
    activation=None                 # Логиты без активации (для стабильности лосса)
)
print("Модель успешно создана.")


# 5. ТЕСТОВЫЙ ЗАПУСК (FORWARD PASS)
# Берем одну картинку из обучающего датасета
img_tensor, mask_tensor = train_dataset[0]

# Добавляем размерность батча (Batch Dimension), так как сеть ожидает на вход пачку картинок [B, C, H, W]
# Наш тензор [3, 512, 512] превратится в [1, 3, 512, 512]
input_batch = img_tensor.unsqueeze(0)

# Пропускаем через модель (выключаем расчет градиентов для скорости)
with torch.no_grad():
    output = model(input_batch)

print(f"Входной тензор: {input_batch.shape}")
print(f"Выходной тензор модели (предсказание): {output.shape}")
print("Тестовый проход через нейросеть выполнен без ошибок!")