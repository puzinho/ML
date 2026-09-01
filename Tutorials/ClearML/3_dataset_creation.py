# 3_dataset_creation.py

"""
Этот скрипт демонстрирует создание датасета ClearML через Python API.
Датасет будет загружен на сервер ClearML и станет доступен для использования
в экспериментах как локально, так и на удалённых агентах.
"""

import pandas as pd
from clearml import Dataset

# Путь к локальному датасету
DATA_PATH = "./data/synthetic_dataset.csv"

print("=" * 60)
print("Создание датасета ClearML")
print("=" * 60)

# 1. Создаем новый датасет
# dataset_project - проект, в котором будет создан датасет
# dataset_name - имя датасета (можно создавать версии с одинаковым именем)
print("\n Создание нового датасета...")
dataset = Dataset.create(
    dataset_project="Tutorial", dataset_name="Synthetic Dataset"
)

# Добавляем теги для удобной фильтрации
dataset.add_tags(["synthetic", "classification", "tutorial"])

print(f"Датасет создан с ID: {dataset.id}")

# 2. Добавляем файлы в датасет
# Можно добавлять отдельные файлы или целые директории
print("\nДобавление файлов в датасет...")
dataset.add_files(path=DATA_PATH)
print(f"Файл добавлен: {DATA_PATH}")

# 3. Добавляем метаданные и статистику
# Это помогает понять содержимое датасета без его загрузки
print("\nДобавление метаданных...")

# Загружаем датасет для получения статистики
df = pd.read_csv(DATA_PATH)

# Создаем словарь с метаданными
metadata = {
    "description": "Синтетический датасет для бинарной классификации",
    "n_samples": df.shape[0],
    "n_features": df.shape[1] - 1,  # -1 чтобы исключить target
    "target_column": "target",
    "feature_names": list(df.columns[:-1]),
    "class_distribution": df["target"].value_counts().to_dict(),
}

# Записываем метаданные в датасет
dataset.get_logger().report_text(
    f"Dataset shape: {df.shape}\n"
    f"Features: {metadata['n_features']}\n"
    f"Samples: {metadata['n_samples']}\n"
    f"Class distribution: {metadata['class_distribution']}"
)

# Сохраняем базовую статистику
stats = df.describe()
dataset.get_logger().report_table(
    title="Dataset Statistics", series="Summary", table_plot=stats
)

print("Метаданные добавлены")

# 4. Финализируем датасет и загружаем на сервер
# После вызова upload() датасет станет доступен для использования
print("\nЗагрузка датасета на сервер ClearML...")
dataset.upload()
print("Датасет успешно загружен!")

# 5. Финализируем датасет (делаем read-only)
dataset.finalize()
print("Датасет финализирован (read-only)")

# Выводим важную информацию
print("\n" + "=" * 60)
print("ИНФОРМАЦИЯ О ДАТАСЕТЕ")
print("=" * 60)
print(f"Dataset ID: {dataset.id}")
print(f"Dataset Name: {dataset.name}")
print(f"Project: {dataset.project}")
print(f"Number of files: {len(dataset.list_files())}")
print("\n Теперь можешь использовать датасет в экспериментах:")
print("\n   # Рекомендуемый способ - по имени (всегда последняя версия):")
print("   dataset = Dataset.get(")
print(f"       dataset_project='{dataset.project}',")
print(f"       dataset_name='{dataset.name}'")
print("   )")
print("\n   # Альтернативный способ - по конкретному ID версии:")
print(f"   dataset = Dataset.get(dataset_id='{dataset.id}')")
print("=" * 60)

# Дополнительно: создание новой версии датасета
print("\n\n Примечание: Создание новой версии")
print("-" * 60)
print("Если нужно обновить датасет, создай новую версию:")
print(
    """
# Пример создания новой версии
from clearml import Dataset

new_dataset = Dataset.create(
    dataset_project='{project}',
    dataset_name='{name}',
    parent_datasets=['{dataset_id}']  # ID текущей версии как parent
)
new_dataset.add_files(path="./data/updated_dataset.csv")
new_dataset.upload()
new_dataset.finalize()

# Теперь Dataset.get() по имени вернёт новую версию автоматически!
""".format(project=dataset.project, name=dataset.name, dataset_id=dataset.id)
)
print("-" * 60)
