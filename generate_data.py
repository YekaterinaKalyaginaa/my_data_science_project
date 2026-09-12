import pandas as pd
import numpy as np
import os

# Проверяем, существует ли папка, и создаём её, если нет
os.makedirs('data/raw', exist_ok=True)

# Генерируем случайные данные (100 строк, 3 колонки со случайными числами)
df = pd.DataFrame(np.random.randn(100, 3), columns=['Feature1', 'Feature2', 'Feature3'])

# Сохраняем результат в файл my_data.csv
df.to_csv('data/raw/my_data.csv', index=False)
print("Файл data/raw/my_data.csv успешно создан!")
