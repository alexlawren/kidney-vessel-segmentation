import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

# Пути к данным
DATA_DIR = "./data"
POLYGONS_PATH = os.path.join(DATA_DIR, "polygons.jsonl")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")

# 1. Читаем первую строку (изображение 0006ff2aa7cd)
with open(POLYGONS_PATH, 'r') as f:
    first_line = f.readline()
    data = json.loads(first_line)

image_id = data['id']
annotations = data['annotations']

# 2. Создаем абсолютно черное изображение размером 512x512
# Режим 'L' означает одноканальное черно-белое изображение (8-бит)
mask = Image.new('L', (512, 512), 0)
draw = ImageDraw.Draw(mask)

# 3. Рисуем (растеризуем) полигоны сосудов белым цветом (255)
for ann in annotations:
    # Берем ТОЛЬКО сосуды (blood_vessel)
    if ann['type'] == 'blood_vessel':
        for polygon in ann['coordinates']:
            # Переводим координаты в формат кортежей, который понимает Pillow
            flat_poly = [tuple(coord) for coord in polygon]
            # Заливаем область внутри полигона белым цветом (255)
            draw.polygon(flat_poly, fill=255)

# 4. Загружаем оригинальное розовое изображение для сравнения
image_path = os.path.join(TRAIN_IMG_DIR, f"{image_id}.tif")
original_image = Image.open(image_path)

# 5. Выводим две картинки рядом: оригинал и полученную маску
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(original_image)
axes[0].set_title("Оригинальное изображение")
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Сгенерированная маска сосудов (Ground Truth)")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# (Опционально) Сохраняем маску на диск, чтобы посмотреть на нее
# mask.save("test_mask.png")