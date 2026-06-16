import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Локальные пути к данным (папка data лежит в корне проекта)
DATA_DIR = "./data"
POLYGONS_PATH = os.path.join(DATA_DIR, "polygons.jsonl")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")

# 1. Читаем самую первую строку из файла разметки
with open(POLYGONS_PATH, 'r') as f:
    first_line = f.readline()
    data = json.loads(first_line)

image_id = data['id']
annotations = data['annotations']

print(f"ID изображения: {image_id}")
print(f"Количество размеченных объектов на этом снимке: {len(annotations)}")

# 2. Загружаем само изображение по его ID
image_path = os.path.join(TRAIN_IMG_DIR, f"{image_id}.tif")
image = Image.open(image_path)

# 3. Визуализируем картинку и накладываем контуры сосудов
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image)

# Проходим по всем аннотациям на этой картинке
for ann in annotations:
    structure_type = ann['type'] # 'blood_vessel' или 'glomerulus'
    coords = ann['coordinates']  # список точек полигона [[[x1, y1], [x2, y2], ...]]
    
    # Цвет контура в зависимости от типа структуры
    color = 'red' if structure_type == 'blood_vessel' else 'blue'
    
    for polygon in coords:
        # Рисуем полигон поверх картинки
        poly_patch = patches.Polygon(polygon, closed=True, edgecolor=color, facecolor='none', linewidth=2)
        ax.add_patch(poly_patch)

plt.title(f"Изображение {image_id} (Красный - сосуды, Синий - клубочки)")
plt.axis('off')
plt.show()