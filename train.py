import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import albumentations as A
import segmentation_models_pytorch as smp
from dataset import KidneyDataset

# 1. ОСНОВНЫЕ НАСТРОЙКИ И ГИПЕРПАРАМЕТРЫ СТАБИЛИЗАЦИИ
DATA_DIR = "./data"
BATCH_SIZE = 8          
GRADIENT_ACCUMULATION_STEPS = 2  # Виртуально увеличиваем батч до 16 (8 * 2)
EPOCHS = 175                     # Настроено на глубокое обучение в 175 эпох

# Сбалансированный дифференциальный шаг для долгой дистанции
LR_DECODER = 2e-4       # Обычный шаг для нового декодера U-Net
LR_ENCODER = 2e-5       # Микро-шаг для сохранения знаний предобученного ResNet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство для вычислений: {device}")


# 2. ПОДГОТОВКА ДАННЫХ И РАЗДЕЛЕНИЕ (WSI SPLIT)
tile_meta = pd.read_csv(os.path.join(DATA_DIR, "tile_meta.csv"))

annotated_ids = set()
with open(os.path.join(DATA_DIR, "polygons.jsonl"), 'r') as f:
    for line in f:
        data = json.loads(line)
        annotated_ids.add(data['id'])

annotated_meta = tile_meta[tile_meta['id'].isin(annotated_ids)].copy()

# Обучаем на почках 1, 2, 3. Проверяем на почке 4.
train_ids = annotated_meta[annotated_meta['source_wsi'].isin([1, 2, 3])]['id'].tolist()
val_ids = annotated_meta[annotated_meta['source_wsi'] == 4]['id'].tolist()


# 3. АУГМЕНТАЦИИ И ДАТАЛОАДЕРЫ (С ЧЕМПИОНСКИМ МАСШТАБОМ)
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    # Масштабирование от 20% до 200% (scale_limit=-0.8 ... 1.0) и повороты на 180 градусов
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.8, 1.0), rotate_limit=180, p=0.8),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

train_dataset = KidneyDataset(data_dir=DATA_DIR, image_ids=train_ids, transform=train_transform)
val_dataset = KidneyDataset(data_dir=DATA_DIR, image_ids=val_ids, transform=None)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# 4. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ U-NET (SMP)
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)
model.to(device)


# 5. НАСТРОЙКА ДИФФЕРЕНЦИАЛЬНОГО ОПТИМИЗАТОРА И ПЛАНИРОВЩИКА
encoder_params = model.encoder.parameters()
decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]

# Задаем жесткую регуляризацию weight_decay=1e-2 (0.01), как у лидера
optimizer = optim.AdamW([
    {"params": encoder_params, "lr": LR_ENCODER},
    {"params": decoder_params, "lr": LR_DECODER}
], weight_decay=1e-2)

# Планировщик автоматически адаптируется под 175 эпох
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


# 6. ЛОСС-ФУНКЦИИ И МЕТРИКА DICE
criterion_bce = nn.BCEWithLogitsLoss()
criterion_dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

def calculate_dice(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + 1e-8)


# 7. СТАБИЛИЗИРОВАННЫЙ ЦИКЛ ОБУЧЕНИЯ (TRAINING LOOP)
print("Начало процесса стабилизированного обучения на 175 эпох...")
best_val_dice = 0.0

for epoch in range(EPOCHS):
    # --- ЭТАП ОБУЧЕНИЯ ---
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        outputs = model(images)
        
        # Считаем лосс и делим его на шаги аккумуляции градиента
        loss = 0.5 * criterion_bce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        loss.backward() # Накапливаем градиенты
        
        # Шаг оптимизатора делаем только раз в GRADIENT_ACCUMULATION_STEPS шагов
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Умножаем обратно на GRADIENT_ACCUMULATION_STEPS для честного отображения Train Loss
        train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS * images.size(0)
        
    train_loss = train_loss / len(train_dataset)
    scheduler.step()
    
    # --- ЭТАП ВАЛИДАЦИИ ---
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = 0.5 * criterion_bce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)
            
            val_loss += loss.item() * images.size(0)
            val_dice += calculate_dice(outputs, masks).item() * images.size(0)
            
    val_loss = val_loss / len(val_dataset)
    val_dice = val_dice / len(val_dataset)
    
    # Выводим текущий LR декодера для контроля за планировщиком
    current_lr_dec = optimizer.param_groups[1]['lr']
    
    print(f"Эпоха [{epoch+1}/{EPOCHS}] | Dec LR: {current_lr_dec:.6f} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Dice Score: {val_dice:.4f}")
    
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "best_model.pth")
        print(f"==> Сохранена новая лучшая модель с Val Dice: {best_val_dice:.4f}")

print("\nОбучение завершено!")
print(f"Лучший результат на валидации (Val Dice): {best_val_dice:.4f}")