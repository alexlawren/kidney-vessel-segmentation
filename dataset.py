import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

class KidneyDataset(Dataset):
    def __init__(self, data_dir, image_ids, transform=None):
        """
        Args:
            data_dir (str): Путь к папке data.
            image_ids (list): Список ID картинок, которые мы берем в этот датасет.
            transform: Аугментации из библиотеки Albumentations.
        """
        self.data_dir = data_dir
        self.train_img_dir = os.path.join(data_dir, "train")
        self.transform = transform
        
        # 1. Предзагружаем ВСЮ разметку в память ОДИН раз.
        # Читать файл с диска на каждом шаге обучения слишком медленно.
        self.annotations = {}
        polygons_path = os.path.join(data_dir, "polygons.jsonl")
        with open(polygons_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.annotations[data['id']] = data['annotations']
        
        # 2. Оставляем только те ID картинок, для которых ЕСТЬ разметка в файле.
        # Из EDA мы знаем, что размечено всего 1633 картинки из 7033.
        self.image_ids = [img_id for img_id in image_ids if img_id in self.annotations]

    def __len__(self):
        # Обязательный метод PyTorch: возвращает размер датасета
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Обязательный метод PyTorch: загружает один элемент по индексу
        image_id = self.image_ids[idx]
        
        # 1. Загружаем оригинальную картинку
        image_path = os.path.join(self.train_img_dir, f"{image_id}.tif")
        # Конвертируем в RGB (3 канала), так как tif может прочитаться иначе
        image = Image.open(image_path).convert("RGB")
        image = np.array(image) # Переводим в numpy array для Albumentations
        
        # 2. Генерируем маску "на лету" из полигонов
        mask = Image.new('L', (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        
        for ann in self.annotations[image_id]:
            if ann['type'] == 'blood_vessel':
                for polygon in ann['coordinates']:
                    flat_poly = [tuple(coord) for coord in polygon]
                    draw.polygon(flat_poly, fill=255)
                    
        mask = np.array(mask) # Переводим в numpy array
        
        # 3. Применяем аугментации (если они переданы)
        # Albumentations требует, чтобы маска передавалась отдельно,
        # так как она должна повернуться/отразиться точно так же, как и картинка!
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # 4. Нормализуем данные для нейросети
        # Переводим пиксели картинки из диапазона [0..255] в [0..1] и меняем размерность на (Каналы, Высота, Ширина)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)) # Было (H, W, C) -> стало (C, H, W)
        
        # Маску переводим в диапазон [0..1] и добавляем канал: (1, H, W)
        mask = (mask > 127).astype(np.float32) # Все, что ярче половины, делаем 1.0, остальное 0.0
        mask = np.expand_dims(mask, axis=0) # Стало (1, H, W)
        
        # Возвращаем тензоры PyTorch
        return torch.tensor(image), torch.tensor(mask)