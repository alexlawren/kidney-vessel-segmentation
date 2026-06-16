import pandas as pd
from dataset import KidneyDataset
import matplotlib.pyplot as plt

# Читаем таблицу метаданных картинок, чтобы получить список всех ID
tile_meta = pd.read_csv("./data/tile_meta.csv")
all_ids = tile_meta['id'].tolist()

# Создаем наш датасет
dataset = KidneyDataset(data_dir="./data", image_ids=all_ids)
print(f"Всего размеченных картинок в датасете: {len(dataset)}")

# Берем самый первый элемент из датасета
img_tensor, mask_tensor = dataset[0]

print(f"Размерность тензора картинки: {img_tensor.shape}") # Должно быть (3, 512, 512)
print(f"Размерность тензора маски: {mask_tensor.shape}")     # Должно быть (1, 512, 512)

# Визуализируем, чтобы убедиться, что тензоры правильные
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# Для отрисовки возвращаем размерность обратно в (H, W, C)
plt.imshow(img_tensor.permute(1, 2, 0).numpy())
plt.title("Картинка из датасета")

plt.subplot(1, 2, 2)
plt.imshow(mask_tensor.squeeze().numpy(), cmap='gray')
plt.title("Маска из датасета")
plt.show()