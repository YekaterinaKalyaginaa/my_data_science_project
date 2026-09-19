# Лабораторная работа 1: Введение и воспроизводимость исследований

Цель: настроить проект, применить практики воспроизводимости (Git, DVC) и построить конвейер обработки данных с использованием `Pipeline`, `ColumnTransformer` и `OneHotEncoder` / кастомного трансформера.

## Что сделано
- создана структура проекта для Data Science;
- добавлена генерация и сохранение данных в `data/raw/wine_dataset.csv`;
- настроен DVC для отслеживания датасета;
- добавлен пример ML-конвейера с `Pipeline` и `ColumnTransformer`;
- добавлен кастомный трансформер `BinarizeAlcoholTransformer` для бинирования признака `alcohol` по порогу `13.0`;
- используется настоящая целевая переменная `target` из датасета Wine: три сорта вина.

## Структура проекта
- `prepare_wine_dataset.py` — создает датасет из `sklearn.datasets.load_wine()` и сохраняет его в `data/raw/wine_dataset.csv`;
- `src/wine_transformers.py` — пользовательский трансформер;
- `src/train_wine_model.py` — обучение модели и запись метрик;
- `data/raw/` — исходные данные;
- `models/` — артефакты модели;
- `reports/` — метрики в JSON.

## Запуск
1. Установите зависимости:
   `pip install -r requirements.txt`
2. Сгенерируйте датасет:
   `python prepare_wine_dataset.py`
3. Обучите модель:
   `python src/train_wine_model.py`

## Примечание
Для отслеживания датасета DVC можно использовать команду:
`dvc add data/raw/wine_dataset.csv`
