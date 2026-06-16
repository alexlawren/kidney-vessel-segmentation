import pandas as pd

# Загружаем метаданные снимков
tile_meta = pd.read_csv("./data/tile_meta.csv")
polygons_path = "./data/polygons.jsonl"

# Читаем размеченные ID
import json
annotated_ids = set()
with open(polygons_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        annotated_ids.add(data['id'])

# Фильтруем метаданные: оставляем только РАЗМЕЧЕННЫЕ картинки
annotated_meta = tile_meta[tile_meta['id'].isin(annotated_ids)].copy()

# Сгруппируем и посмотрим, сколько картинок у нас приходится на каждую почку (source_wsi)
wsi_distribution = annotated_meta['source_wsi'].value_counts()
print("Распределение размеченных картинок по источникам (WSI):")
print(wsi_distribution)